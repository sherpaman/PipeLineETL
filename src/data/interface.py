import os
import re
import json
import pickle
from typing import Union, Dict
import pathlib

import geopandas
import pandas
import pandas as pd
import xarray


class Interface:
    """Interface Object used to access data from the file-system"""

    def __init__(self, root: str):
        self.root = root

    def __repr__(self):
        out = self.nested_repr(self.scheme, level=0)
        return out

    def _filter_path(self, scheme: dict, match: str, parent: str = None) -> \
            dict:
        """
        Parameters
        ----------
        scheme: dict
            dictionary (possibly nested) representing a directory/file
        match: str
            regular expression used to filter filenames

        Returns
        -------
        out: dict
            subset of the scheme dictionary with full-path of file included
        """
        if parent is None:
            parent = self.root
        m = re.compile(match)
        temp = {}
        for k, v in scheme.items():
            if 'files' in scheme[k]:
                temp[k] = {}
                for f in scheme[k]['files']:
                    # This may be improved with pattern matching (3.10)
                    if hit := m.search(f):
                        name = f[:hit.start()]
                        temp[k][name] = os.path.join(parent, k, f)
            else:
                new_parent = os.path.join(parent, k)
                temp[k] = self._filter_path(scheme[k], match, new_parent)
        return temp

    @property
    def data_set_path(self):
        out = self._filter_path(self.scheme, r'_dataset$')
        return out['data']

    @property
    def model_path(self):
        temp = self._filter_path(self.scheme, r'_model_pickle$')
        # regex catching the sting before the first "_" in the filename
        model_re = re.compile(r'([a-zA-Z0-9,-]+)_([a-zA-Z0-9_]+)')
        out = {}
        for k, v in temp['models'].items():
            s = model_re.search(k)
            if s.group(1) in out:
                out[s.group(1)][s.group(2)] = v
            else:
                out[s.group(1)] = {s.group(2): v}
        return out

    @property
    def filter_path(self):
        out = self._filter_path(self.scheme, r'_filter$')
        return out['filters']

    @property
    def staging_path(self):
        out = self._filter_path(self.scheme, r'\.json$')
        return out['staging']

    @property
    def scheme(self) -> dict:
        scheme = self.nested_file_path(self.root)
        return scheme

    @property
    def path(self) -> dict:
        scheme = self.nested_dir_path(self.root)
        return scheme

    def nested_file_path(self,
                         root: str) -> dict:
        """
        Parameters
        ----------
        self
        root: str
            starting directory

        Returns
        -------
        out: dict
            dictionary representing the directory structure starting from root
        """
        out = {}
        files = []
        for d in os.listdir((r := root)):
            if os.path.isdir((p := os.path.join(r, d))):
                out[d] = self.nested_file_path(p)
            else:
                files.append(d)
        if len(files) > 0:
            out['files'] = files
        return out

    def nested_dir_path(self,
                        root: str) -> Union[str, Dict]:
        """
        Parameters
        ----------
        self
        root: str
            starting directory

        Returns
        -------
        out: dict
            dictionary representing the directory structure starting from root
        """
        out = {}
        dirs = [d for d in os.listdir(root) if
                os.path.isdir(os.path.join(root, d))]
        if len(dirs) == 0:
            return root
        for d in dirs:
            if os.path.isdir((p := os.path.join(root, d))):
                out[d] = self.nested_dir_path(p)
        return out

    def nested_repr(self,
                    s: dict,
                    level: int = 0,
                    closing: bool = False) -> str:
        if level != 0:
            out = ''
        else:
            out = self.root + '\n'
        if not closing:
            head = '│   ' * level
        else:
            head = '│   ' * (level - 1) + '    '
        files = []
        dirs = [k for k in s.keys() if (k != 'files')]
        if 'files' in s:
            files = s['files']
        nk = len(dirs)
        nf = len(files)
        for i, v in enumerate(dirs):
            if i < (nk - 1):
                link = '├──'
            else:
                link = '└──'
                closing = True
            out += f'{head}{link}{v}\n'
            out += self.nested_repr(s[v], level=level + 1, closing=closing)
        if nf > 0:
            for i, f in enumerate(files):
                if i < (nf - 1):
                    link = '├─* '
                else:
                    link = '└─* '
                out += f'{head}{link}{f}\n'
        return out

    def get_dataset_path(self, stage: str, name: str) -> pathlib.Path:
        """
        Return path of the corresponding data-set

        Parameters
        ----------
        stage: str
            {'raw', 'interim', 'processed', 'external'}
        name : str

        Returns
        -------
        out: pathlib.Path

        """
        base_dir = pathlib.Path(self.path['data'][stage])
        out_path = base_dir.joinpath(f'{name}_dataset')
        return out_path

    def get_dataset(self,
                    stage: str,
                    name: str
                    ) -> Dict[str, Union[pandas.DataFrame,
                                         geopandas.GeoDataFrame]]:
        """
        Reads dataset from the file-system and returns a pandas
        DataFrame object.

        Parameters
        ----------

        stage: str
            {'raw', 'interim', 'processed', 'external'}
        name : str

        Returns
        -------

        out: dict[str: pandas.DataFrame]

        Raises
        ------
        ValueError: when specified type does not match the options
        """

        path = self.get_dataset_path(stage, name)
        if path.is_file():
            with open(path, 'rb') as fi:
                out = pickle.load(fi)
        else:
            raise FileNotFoundError(path)
        return out

    def save_dataset(self,
                     dataset: Dict[str, Union[pandas.DataFrame,
                                              geopandas.GeoDataFrame]],
                     stage: str,
                     name: str) -> None:
        """
        Reads dataset from the file-system and returns a pandas
        DataFrame object.

        Parameters
        ----------
        dataset : dict[str: pandas.DataFrame]
            data to be saved
        stage: str
            {'raw', 'interim', 'processed', 'external'}
        name : str

        Raises
        ------
        ValueError: when specified type does not match the options
        """
        path = self.get_dataset_path(stage, name)
        with open(path, 'wb') as fo:
            pickle.dump(dataset, fo)
        return

    def get_filter_history_path(self, name: str) -> pathlib.Path:
        base_dir = pathlib.Path(self.path['filters'])
        out_path = base_dir.joinpath(f'{name}_filter_history')
        return out_path

    def save_filter_history(self,
                            filter_history,
                            name: str,
                            force: bool = False) -> None:
        out_file = self.get_filter_history_path(name=name)
        if out_file.exists() & (not force):
            print(f'{out_file} already present')
            raise FileExistsError
        else:
            with open(out_file, 'wb') as fo:
                pickle.dump(filter_history, fo)
        return

    def get_filter_history(self, name: str) -> Dict[int, xarray.DataArray]:
        input_file = self.get_filter_history_path(name)
        if input_file.exists():
            with open(input_file, 'rb') as fi:
                filter_history = pickle.load(fi)
        else:
            raise FileNotFoundError(input_file)
        return filter_history

    def get_filter_path(self, name: str) -> pathlib.Path:
        base_dir = pathlib.Path(self.path['filters'])
        out_path = base_dir.joinpath(f'{name}_filter')
        return out_path

    def save_filter(self,
                    f_data: pandas.DataFrame,
                    name: str,
                    force: bool = False):
        out_file = self.get_filter_path(name=name)
        if out_file.exists() & (not force):
            print(f'{out_file} already present')
            raise FileExistsError
        else:
            f_data.to_parquet(out_file)
        return

    def get_filter(self, name: str):
        input_file = self.get_filter_path(name=name)
        if input_file.exists():
            out = pd.read_parquet(input_file)
        else:
            raise FileNotFoundError(input_file)
        return out

    def get_staging_path(self, stage: str, name: str):
        base_dir = pathlib.Path(self.path['staging'][stage])
        out_file = base_dir.joinpath(f'{name}.json')
        return out_file

    def get_staging_file(self, stage: str, name: str) -> json:
        """
        Reads json file from the file-system and returns its data.

        Parameters
        ----------

        name: str
        stage: {'edit', 'logs', 'root'}

        Returns
        -------
        new: json object

        Raises
        ------
        ValueError: when specified type does not match the options
        """

        stages = list(self.staging_path.keys())
        if stage not in self.staging_path:
            raise ValueError(f'"stage" must be in {stages}')
        else:
            path = self.staging_path[stage][name]

        with open(path) as f:
            data = json.load(f)
        return data

    def directory_structure(self, path: str) -> str:
        """
        Create string representation of directory structure with its contents
        from an specified path.

        Parameters
        ----------
        path: str

        Returns
        -------
        structure: str
        """
        if os.path.exists(path):
            explore = self.nested_file_path(path)
            structure = self.nested_repr(explore)
        else:
            raise ValueError(f'path : {path} is not valid')
        return structure
