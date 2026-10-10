#!/usr/bin/env python3
"""Unpack a ScanNet .sens blob into the per-frame layout SplaTAM expects.

Writes <out>/color/<i>.jpg, <out>/depth/<i>.png, <out>/pose/<i>.txt and
<out>/intrinsic/*.txt, matching what ScanNet's own SensReader produces so
SplaTAM's ScannetDataset globs pick it up unchanged.

WHY NOT ScanNet/SensReader/python. That reader is Python 2 -- `print` is a
statement there, so it does not even import on py3 -- and two of its hot
paths are badly shaped besides:

  - It reads frame payloads with ''.join(struct.unpack('c'*n, ...)), which
    allocates one bytes object per byte. On py3 that also raises outright,
    since the tuple holds bytes while the separator is str.
  - It loads every frame into memory before writing anything, so a 1.1 GB
    scan becomes several GB of live objects.
  - It re-encodes each JPEG through imageio, losing quality to produce a file
    the input already contained verbatim, and writes 16-bit depth PNGs
    through pypng row lists.

The binary layout below is theirs, field for field. Everything else is not.

Requires: numpy, opencv-python (both already in the SLAM environments).
"""

import argparse
import os
import struct
import sys
import zlib

import numpy as np

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort',
                          2: 'occi_ushort'}


def _save_mat(matrix, path):
    # Same on-disk shape as SensorData.save_mat_to_file: four rows, '%f'.
    with open(path, 'w') as f:
        np.savetxt(f, matrix, fmt='%f')


def convert(sens_path, out_dir, frame_skip=1):
    color_dir = os.path.join(out_dir, 'color')
    depth_dir = os.path.join(out_dir, 'depth')
    pose_dir = os.path.join(out_dir, 'pose')
    intr_dir = os.path.join(out_dir, 'intrinsic')
    for d in (color_dir, depth_dir, pose_dir, intr_dir):
        os.makedirs(d, exist_ok=True)

    import cv2  # imported late so --help works without opencv present

    with open(sens_path, 'rb') as f:
        version = struct.unpack('I', f.read(4))[0]
        if version != 4:
            raise ValueError('unsupported .sens version %d, expected 4' % version)

        strlen = struct.unpack('Q', f.read(8))[0]
        sensor_name = f.read(strlen).decode('utf-8', 'replace')

        def mat4():
            return np.asarray(struct.unpack('f' * 16, f.read(64)),
                              dtype=np.float32).reshape(4, 4)

        intrinsic_color, extrinsic_color = mat4(), mat4()
        intrinsic_depth, extrinsic_depth = mat4(), mat4()

        color_compression = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
        depth_compression = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
        color_width = struct.unpack('I', f.read(4))[0]
        color_height = struct.unpack('I', f.read(4))[0]
        depth_width = struct.unpack('I', f.read(4))[0]
        depth_height = struct.unpack('I', f.read(4))[0]
        depth_shift = struct.unpack('f', f.read(4))[0]
        num_frames = struct.unpack('Q', f.read(8))[0]

        print('sensor : %s' % sensor_name)
        print('frames : %d' % num_frames)
        print('color  : %dx%d (%s)' % (color_width, color_height, color_compression))
        print('depth  : %dx%d (%s), shift %g'
              % (depth_width, depth_height, depth_compression, depth_shift))

        if color_compression != 'jpeg':
            raise ValueError('color compression %r not handled; this scan is '
                             'not stored as jpeg' % color_compression)
        if depth_compression != 'zlib_ushort':
            raise ValueError('depth compression %r not handled' % depth_compression)

        _save_mat(intrinsic_color, os.path.join(intr_dir, 'intrinsic_color.txt'))
        _save_mat(extrinsic_color, os.path.join(intr_dir, 'extrinsic_color.txt'))
        _save_mat(intrinsic_depth, os.path.join(intr_dir, 'intrinsic_depth.txt'))
        _save_mat(extrinsic_depth, os.path.join(intr_dir, 'extrinsic_depth.txt'))

        n_written = 0
        n_bad_pose = 0
        for i in range(num_frames):
            camera_to_world = np.asarray(struct.unpack('f' * 16, f.read(64)),
                                         dtype=np.float32).reshape(4, 4)
            f.read(8)   # timestamp_color, unused
            f.read(8)   # timestamp_depth, unused
            color_bytes = struct.unpack('Q', f.read(8))[0]
            depth_bytes = struct.unpack('Q', f.read(8))[0]
            color_data = f.read(color_bytes)
            depth_data = f.read(depth_bytes)

            if len(color_data) != color_bytes or len(depth_data) != depth_bytes:
                raise EOFError('truncated .sens at frame %d: the download is '
                               'incomplete, re-run download_scannet.sh' % i)

            if i % frame_skip != 0:
                continue

            # ScanNet marks frames whose pose failed to solve with -inf. Kept
            # as-is rather than dropped: the dataloader zips color/depth/pose
            # by index, so removing one would shift every later frame.
            if not np.isfinite(camera_to_world).all():
                n_bad_pose += 1

            # The payload already IS a jpeg. Writing it verbatim is lossless
            # and avoids a decode/re-encode round trip per frame.
            with open(os.path.join(color_dir, '%d.jpg' % i), 'wb') as out:
                out.write(color_data)

            depth = np.frombuffer(zlib.decompress(depth_data),
                                  dtype=np.uint16).reshape(depth_height, depth_width)
            # cv2 writes 16-bit single-channel PNG natively, which is what
            # SplaTAM's loader divides by png_depth_scale.
            cv2.imwrite(os.path.join(depth_dir, '%d.png' % i), depth)

            _save_mat(camera_to_world, os.path.join(pose_dir, '%d.txt' % i))

            n_written += 1
            if n_written % 200 == 0:
                print('  %d frames ...' % n_written, flush=True)

    print('wrote %d frames to %s' % (n_written, out_dir))
    if n_bad_pose:
        print('NOTE: %d of %d poses are non-finite (-inf). ScanNet ships these '
              'where tracking failed; they are kept so frame indices stay '
              'aligned, but they will corrupt an ATE that averages over them.'
              % (n_bad_pose, n_written))
    return n_written


def main():
    ap = argparse.ArgumentParser(
        description='Unpack a ScanNet .sens into color/depth/pose/intrinsic.')
    ap.add_argument('--filename', required=True, help='path to the .sens blob')
    ap.add_argument('--output_path', required=True, help='scene directory to write')
    ap.add_argument('--frame_skip', type=int, default=1, help='keep every Nth frame')
    args = ap.parse_args()

    if not os.path.isfile(args.filename):
        sys.exit('no such file: %s' % args.filename)
    convert(args.filename, args.output_path, args.frame_skip)


if __name__ == '__main__':
    main()
