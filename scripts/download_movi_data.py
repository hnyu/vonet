import tensorflow_datasets as tfds
import os
import sys
import pickle
import numpy as np

"""Usage: python download_movi_data.py movi_a
"""

HOME = os.path.expanduser('~')


def _download(name, out_dir):
    ds = tfds.load(name + "/128x128", data_dir="gs://kubric-public/tfds")

    def _write_videos(split):
        data = ds.get(split, None)
        if data is None:
            return
        data = tfds.as_numpy(data)
        print("====Total videos of %s: %d====" % (split, len(data)))
        output_dir = os.path.join(out_dir, name, split)
        os.system("mkdir -p %s" % output_dir)
        print("dataset size: ", len(data))
        for i, video in enumerate(data):
            output_path = os.path.join(output_dir, "video_%s.pkl" % video['metadata']['video_name'].decode())
            try:
                with open(output_path, 'rb') as f:
                    pickle.load(f)
                continue
            except:
                pass

            print("Pickling %s" % output_path)
            keys = ['video', 'depth', 'segmentations', 'metadata']
            video_ = {k: video[k] for k in keys}
            if "flows" in video:
                video_['flow'] = video['flows']
            if "backward_flow" in video:
                video_['flow'] = video["backward_flow"]
                #video_['metadata']['flow_range'] = video['metadata'][
                #    'backward_flow_range']
            assert 'flow' in video_
            with open(output_path, 'wb') as f:
                pickle.dump(video_, f)

    for split in ['train', 'validation']:
        _write_videos(split)

name = sys.argv[1]
_download(name, os.path.join(HOME, 'movi/pickled'))
