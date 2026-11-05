"""Generate noisy labels for train only, or for train + val in ckpt=1 mode.

Concept noise flips binary labels; class noise replaces labels with a different
class. Images and other metadata are preserved. Global noise samples a fixed
count of entries. CPO noise flips a fixed count per image using a path-derived seed.
Test labels always remain clean. With ckpt=0, validation labels also remain clean.
"""

import os
import copy
import json
import random
import pickle
import argparse
import shutil
from pathlib import Path

import numpy as np

def add_concept_noise(
    out_dir, data_dir, noise_rate, n_concepts=112, n_classes=200,
    n_images=None, datasets=('train',),
):
    """
    Add independent concept noise 
    """
    if n_images is None:
        n_images = get_data_len(data_dir, datasets)

    # Sample an exact global count of concept labels, not a count of images.
    n_attrs = n_concepts * n_images
    n_noise = int(n_concepts * n_images * noise_rate)
    narray = np.concatenate((np.zeros(n_attrs - n_noise), np.ones(n_noise)))
    random.shuffle(narray)
    noise = narray.reshape((n_concepts, n_images))

    print("Concept noise array:", noise)
    print(noise.shape)

    def noise_fn(d, i):
        d['attribute_label'] = [int(not elem) if noise[j, i] else elem
                                for j, elem in enumerate(d['attribute_label'])]
        return d

    return create_new_dataset(out_dir, noise_fn, datasets=datasets, data_dir=data_dir)


def cpo_image_identifier(stored_path, image_dir):
    """Use the local image path, as returned by the CPO CUB DataLoader."""
    stored_path = Path(stored_path)
    if 'images' in stored_path.parts:
        images_index = stored_path.parts.index('images')
        relative_path = Path(*stored_path.parts[images_index + 1:])
    elif stored_path.is_absolute():
        relative_path = Path(*stored_path.parts[-2:])
    else:
        relative_path = stored_path
    return str(Path(image_dir).expanduser().resolve() / relative_path)


def add_cpo_concept_noise(
    out_dir, data_dir, noise_rate, image_dir, seed=24,
    n_concepts=112, datasets=('train',),
):
    """Match CPO's per-image randperm and sum(ord(path)) + seed rule.

    Reference: Emilianopp/ConceptPreferenceOptimization, commit de4d517.
    Use a local CPU generator to reproduce its mask without changing global RNGs.
    Equal character sums intentionally share masks, matching the reference code.
    """
    import torch

    flips_per_image = int(noise_rate * n_concepts)

    def noise_fn(record, sample_index):
        identifier = cpo_image_identifier(record['img_path'], image_dir)
        sample_seed = sum(ord(char) for char in identifier) + seed
        generator = torch.Generator(device='cpu').manual_seed(sample_seed)
        indices = torch.randperm(n_concepts, generator=generator)[:flips_per_image]

        labels = list(record['attribute_label'])
        for concept_index in indices.tolist():
            labels[concept_index] = 1 - labels[concept_index]
        record['attribute_label'] = labels
        return record

    print(f'CPO noise: {flips_per_image}/{n_concepts} concepts per image')
    return create_new_dataset(out_dir, noise_fn, datasets=datasets, data_dir=data_dir)


def add_class_noise(out_dir, data_dir, noise_rate, n_class=200, n_images=None, datasets=('train',)):
    """
    Add independent class noise 
    """
    if n_images is None:
        n_images = get_data_len(data_dir, datasets)

    n_noise = int(n_images * noise_rate)
    noise = np.concatenate((np.zeros(n_images - n_noise), np.ones(n_noise)))
    random.shuffle(noise)

    print("Class noise array:", noise)
    print(noise.shape)

    def noise_fn(d, i):
        if noise[i]:
            original_label = d['class_label']
            new_label = original_label
            while new_label == original_label:
                new_label = random.randrange(0, n_class)
            d['class_label'] = new_label
        return d

    return create_new_dataset(out_dir, noise_fn, datasets=datasets, data_dir=data_dir)


def add_both_noise(
    out_dir, data_dir, noise_rate, n_concepts=112, n_class=200,
    n_images=None, datasets=('train',),
):
    """
    Add independent concept & class noise
    """
    if n_images is None:
        n_images = get_data_len(data_dir, datasets)

    # Calculate the total number of attributes and the number of noisy attributes
    n_attrs = n_concepts * n_images
    n_concept_noise = int(n_concepts * n_images * noise_rate)
    n_class_noise = int(n_images * noise_rate)

    # Generate concept noise
    concept_noise_array = np.concatenate((np.zeros(n_attrs - n_concept_noise), np.ones(n_concept_noise)))
    random.shuffle(concept_noise_array)
    concept_noise = concept_noise_array.reshape((n_concepts, n_images))

    # Generate class noise
    class_noise_array = np.concatenate((np.zeros(n_images - n_class_noise), np.ones(n_class_noise)))
    random.shuffle(class_noise_array)
    class_noise = class_noise_array

    print("Concept noise:", concept_noise)
    print(concept_noise.shape)
    print("Class noise:", class_noise)
    print(class_noise.shape)

    def noise_fn(d, i):
        # Apply concept noise
        d['attribute_label'] = [int(not elem) if concept_noise[j, i] else elem
                                for j, elem in enumerate(d['attribute_label'])]

        # Apply class noise
        if class_noise[i]:
            original_label = d['class_label']
            new_label = original_label
            while new_label == original_label:
                new_label = random.randrange(0, n_class)
            d['class_label'] = new_label
        
        return d

    return create_new_dataset(out_dir, noise_fn, datasets=datasets, data_dir=data_dir)


def create_new_dataset(out_dir, compute_fn, datasets=('train',), data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/add one field of the metadata in each dataset based on compute_fn
    and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
    and return the updated metadata object
    """    
    os.makedirs(out_dir, exist_ok=True)
    statistics = {}
    sample_offset = 0

    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        if not data:
            raise ValueError(f"Empty annotation file: {path}")
        new_data = []

        attribute_changed = 0
        class_changed = 0
        
        for i, d in enumerate(data):
            original_attribute = d['attribute_label']
            original_class = d['class_label']

            new_d = copy.deepcopy(d)
            # Use a global index into the shared mask across train and val.
            new_d = compute_fn(new_d, sample_offset + i)
            new_data.append(new_d)

            if original_class != new_d['class_label']:
                class_changed += 1

            for j in range(len(original_attribute)):
                if original_attribute[j] != new_d['attribute_label'][j]:
                    attribute_changed += 1
        
        print("[Statistics]")
        print("Split : ", dataset)
        print("Dataset size : ", len(data))
        print("Attribute changed : ", attribute_changed, " / ", attribute_changed / (len(data) * len(data[0]['attribute_label'])))
        print("Class changed : ", class_changed, " / ", class_changed / len(data))
        
        with open(os.path.join(out_dir, dataset + '.pkl'), 'wb') as f:
            pickle.dump(new_data, f)

        statistics[dataset] = {
            'samples': len(data),
            'concept_entries': len(data) * len(data[0]['attribute_label']),
            'concept_changed': attribute_changed,
            'class_changed': class_changed,
        }
        sample_offset += len(data)

    return statistics


def get_data_len(data_dir='', datasets=('train',)):
    """Count all samples covered by the shared noise mask."""
    total = 0
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        if not data:
            raise ValueError(f"Empty annotation file: {path}")
        total += len(data)
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp', choices=['concept', 'class', 'both'], required=True)
    parser.add_argument('--out_dir', '--out-dir', required=True, help='Output directory')
    parser.add_argument('--data_dir', '--data-dir', required=True, help='Clean annotation directory')
    parser.add_argument('--noise_rate', '--noise-rate', type=float, required=True)
    parser.add_argument('--n_concepts', '--n-concepts', type=int, default=112)
    parser.add_argument('--n_classes', '--n-classes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument(
        '--concept-noise-mode', choices=['global', 'cpo'], default='global',
        help='global: fixed total flip count; cpo: fixed per-image count with path-derived seeds',
    )
    parser.add_argument('--image-dir', help='Local CUB images root used for CPO image identifiers')
    parser.add_argument(
        '--ckpt', type=int, choices=[0, 1], default=0,
        help='0: noise on train only; 1: noise across train + val for merged-split training',
    )
    args = parser.parse_args()

    if not 0 <= args.noise_rate <= 1:
        parser.error('noise_rate must be between 0 and 1')
    if args.n_concepts < 1 or args.n_classes < 2:
        parser.error('n_concepts must be positive and n_classes must be at least 2')
    if args.concept_noise_mode == 'cpo':
        if args.exp != 'concept':
            parser.error('CPO noise mode supports --exp concept only')
        if args.image_dir is None or not Path(args.image_dir).expanduser().is_dir():
            parser.error('CPO noise mode requires --image-dir pointing to the local images root')

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir == data_dir:
        parser.error('Output directory must differ from the clean data directory')

    for split in ('train', 'val', 'test'):
        if not (data_dir / f'{split}.pkl').is_file():
            parser.error(f'Missing source split: {data_dir / (split + ".pkl")}')
        if (out_dir / f'{split}.pkl').exists():
            parser.error(f'Output already exists: {out_dir / (split + ".pkl")}')
    if (out_dir / 'noise_config.json').exists():
        parser.error('noise_config.json already exists in the output directory')

    noisy_splits = ('train', 'val') if args.ckpt else ('train',)
    dataset_size = 0
    cpo_sample_seeds = set()
    for split in noisy_splits:
        with (data_dir / f'{split}.pkl').open('rb') as file:
            records = pickle.load(file)
        if not records:
            parser.error(f'{split} annotations must not be empty')
        labels = np.asarray([record['attribute_label'] for record in records])
        if labels.shape != (len(records), args.n_concepts):
            parser.error(f'{split} concept labels must have shape [N, n_concepts]')
        if not np.isin(labels, [0, 1]).all():
            parser.error('Concept noise requires binary concept labels')
        if any(not 0 <= record['class_label'] < args.n_classes for record in records):
            parser.error(f'{split} class labels must be in [0, n_classes)')
        if args.concept_noise_mode == 'cpo':
            for record in records:
                identifier = cpo_image_identifier(record['img_path'], args.image_dir)
                cpo_sample_seeds.add(sum(ord(char) for char in identifier) + args.seed)
        dataset_size += len(records)
        del records, labels
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.concept_noise_mode == 'cpo':
        statistics = add_cpo_concept_noise(
            out_dir, data_dir, args.noise_rate, args.image_dir,
            seed=args.seed, n_concepts=args.n_concepts, datasets=noisy_splits,
        )
    elif args.exp == 'concept':
        statistics = add_concept_noise(
            out_dir, data_dir, args.noise_rate,
            n_concepts=args.n_concepts, n_images=dataset_size, datasets=noisy_splits,
        )
    elif args.exp == 'class':
        statistics = add_class_noise(
            out_dir, data_dir, args.noise_rate,
            n_class=args.n_classes, n_images=dataset_size, datasets=noisy_splits,
        )
    else:
        statistics = add_both_noise(
            out_dir, data_dir, args.noise_rate,
            n_concepts=args.n_concepts, n_class=args.n_classes,
            n_images=dataset_size, datasets=noisy_splits,
        )

    # Copy only the untouched splits; never overwrite the noisy val in ckpt=1.
    for split in ('val', 'test'):
        if split not in noisy_splits:
            shutil.copy2(data_dir / f'{split}.pkl', out_dir / f'{split}.pkl')
            print(f'Copied clean {split}.pkl')

    combined_statistics = {
        key: sum(split_statistics[key] for split_statistics in statistics.values())
        for key in ('samples', 'concept_entries', 'concept_changed', 'class_changed')
    }
    print('[Combined statistics]', combined_statistics)

    metadata = {
        'exp': args.exp,
        'noise_rate': args.noise_rate,
        'seed': args.seed,
        'source_dir': str(data_dir),
        'n_concepts': args.n_concepts,
        'n_classes': args.n_classes,
        'ckpt': args.ckpt,
        'noisy_splits': list(noisy_splits),
        'statistics': statistics,
        'combined_statistics': combined_statistics,
        'concept_noise_mode': args.concept_noise_mode,
        'realized_concept_noise_rate': (
            combined_statistics['concept_changed'] / combined_statistics['concept_entries']
        ),
    }
    if args.concept_noise_mode == 'cpo':
        metadata.update({
            'image_dir': str(Path(args.image_dir).expanduser().resolve()),
            'flips_per_image': int(args.noise_rate * args.n_concepts),
            'sample_seed_rule': 'sum(ord(char) for char in local_image_path) + seed',
            'unique_sample_seeds': len(cpo_sample_seeds),
            'reference_commit': 'de4d51738b7d18935255ca9690b1c7d5d36fa05e',
        })
    with (out_dir / 'noise_config.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)
    print(f'Saved noisy dataset to: {out_dir}')
