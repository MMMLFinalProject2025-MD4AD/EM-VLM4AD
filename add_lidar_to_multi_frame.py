import json
import os
import argparse
from nuscenes.nuscenes import NuScenes

def extract_timestamp_from_filename(path):
    # E.g., ...__CAM_FRONT__1533107514412460.jpg
    return int(path.split('__')[-1].replace('.jpg', ''))

def build_sample_token_lookup(nusc):
    # Map (channel, timestamp) → sample_data_token
    lookup = {}
    for sd in nusc.sample_data:
        if sd['fileformat'] == 'jpg' and 'CAM' in sd['channel']:
            lookup[(sd['channel'], sd['timestamp'])] = sd['sample_token']
    return lookup

def build_lidar_token_lookup(nusc):
    # Map sample_token → LIDAR_TOP filename
    sample_to_lidar = {}
    for sample in nusc.sample:
        lidar_token = sample['data']['LIDAR_TOP']
        sd = nusc.get('sample_data', lidar_token)
        #sample_to_lidar[sample['token']] = os.path.join(nusc.dataroot, sd['filename'])
        sample_to_lidar[sample['token']] = os.path.join("data/nuscenes", sd['filename'])
    return sample_to_lidar

def process_json(input_json, output_json, nusc):
    with open(input_json, 'r') as f:
        data = json.load(f)

    sample_token_lookup = build_sample_token_lookup(nusc)
    lidar_lookup = build_lidar_token_lookup(nusc)

    updated_data = []
    for qa, cam_dict in data:
        cam_front_path = cam_dict["CAM_FRONT"]
        timestamp = extract_timestamp_from_filename(cam_front_path)

        # Find matching sample_token using (channel, timestamp)
        sample_token = sample_token_lookup.get(("CAM_FRONT", timestamp), None)
        lidar_path = lidar_lookup.get(sample_token, None)

        if not lidar_path:
            print(f"Warning: No LIDAR_TOP found for {cam_front_path}")
            continue

        # Add to entry
        cam_dict["LIDAR_TOP"] = lidar_path
        updated_data.append([qa, cam_dict])

    # Save updated version
    with open(output_json, 'w') as f:
        json.dump(updated_data, f, indent=2)
    print(f"Saved updated file to {output_json}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True, help="Input QA + multi-view JSON")
    parser.add_argument('--output_json', required=True, help="Path to save modified JSON")
    parser.add_argument('--nuscenes_dir', required=True, help="Path to nuScenes dataset root")
    parser.add_argument('--version', default='v1.0-trainval')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_dir, verbose=False)
    process_json(args.input_json, args.output_json, nusc)

if __name__ == "__main__":
    main()
