import os
import pathlib
import re
import json
from datetime import timedelta, datetime
from timeit import default_timer as timer
from functools import cached_property
from typing import Union, List, Any

import matplotlib.pyplot as plt
import networkx
import pandas
import pandas as pd
import geopandas
import geopandas as gp
import numpy as np
import networkx as nx
import sqlalchemy.engine
from decouple import config
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.exc import TimeoutError

from .querybuild import QueryBuilder
from . import interface

root_dir = ''
Interface = interface.Interface(root=root_dir)


class WasteheroDA:
    """
    Parent Class for WasteheroStaging and WasteheroUpdate.
    """

    def __init__(self,
                 db_url: Union[str, sqlalchemy.engine.Engine],
                 local: interface.Interface = Interface,
                 debug: bool = False):
        self.debug = debug
        self.engine = db_url
        self.local = local

    def __setattr__(self, key, value):
        if key == 'engine':
            self.__dict__[key] = self._create_engine(value)
        else:
            self.__dict__[key] = value

    @staticmethod
    def _get_days_before(days_back: int = 30) -> str:
        """
        Returns date from current to 30 days back. Used for short-update.
        """

        return (datetime.today() -
                timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M')

    def _create_engine(self, url: str) -> sqlalchemy.engine.Engine:
        """Creates SQLAlchemy engine."""
        if self.debug:
            # engine = create_engine(url, echo=True)
            engine = create_engine(url)
        else:
            engine = create_engine(url)

        engine.metadata = MetaData(bind=engine)

        return engine


class WasteheroStaging(WasteheroDA):
    """Class for getting DB schema."""

    def __init__(self,
                 db_url: str,
                 local: interface.Interface = Interface,
                 debug: bool = False):
        super(WasteheroStaging, self).__init__(db_url, local, debug)
        self.inspector = inspect(self.engine)

    @cached_property
    def db_tables(self) -> List:
        """Gets all the tables from WasteHero database as a list()."""

        return sorted([table for table in
                       self.inspector.get_table_names(schema='public')])

    @cached_property
    def db_fields(self) -> dict:
        """
        Given a table_name, returns column info as a list of dicts:

        table_name, [{name, type, nullable, default, autoincrement, ...}]

        (e.g.) analytics_filllevelmeasurement,
                [{
                    name: id,
                    type: int(),
                    ...}]
        """

        return {table: self.inspector.get_columns(table_name=table)
                for table in self.db_tables}

    @cached_property
    def db_foreign_keys(self) -> dict:
        """
        Given a table_name, returns foreign key info as a list of dicts:

        table_name, [{constrained_columns, referred_schema,
                    referred_table, referred_columns, name}]

        (e.g.) analytics_filllevelmeasurement,
                [{
                    constrained_columns: container_device_id,
                    referred_schema: None,
                    referred_table: device_devicetocontainer,
                    referred_columns: id,
                    ...},
                ]
        """

        return {table: self.inspector.get_foreign_keys(table_name=table)
                for table in self.db_tables}

    @cached_property
    def table_fields(self) -> dict:
        """
        Adds extra info to 'db_fields': change field name (out_name:
        str()) or use field (use: bool).
        """

        return {table: {v['name']: {'out_name': None, 'use': True}
                        for v in field}
                for table, field in self.db_fields.items()}

    @cached_property
    def table_foreign_keys_old(self) -> dict:
        """
        Creates dict of dicts as:

        table_name, {referred_table {
                        constrained_column: id,
                        referred_column: id}
                    }
        (e.g.) "analytics_filllevelmeasurement": {
                    "device_devicetocontainer": {
                        "constrained_columns": [
                            "container_device_id"
                        ],
                        "referred_columns": [
                            "id"
                        ]
                    }
        """

        fk_info = ['constrained_columns', 'referred_columns']
        return {t: {f['referred_table']: {info: f[info] for info in fk_info}
                    for f in v
                    if
                    self.table_fields[t][f['constrained_columns'][0]]['use']}
                for t, v in self.db_foreign_keys.items()}

    @property
    def table_foreign_keys(self) -> dict:
        """
        Creates dict of dicts as:

        table_name, {referred_table {
                        constrained_column: id,
                        referred_column: id}
                    }
        (e.g.) "analytics_filllevelmeasurement": {
                    "device_devicetocontainer": {
                        "constrained_columns": [
                            "container_device_id"
                        ],
                        "referred_columns": [
                            "id"
                        ]
                    }
        """

        out = {}
        for t, fts in self.db_foreign_keys.items():
            temp = {}
            for ft in fts:
                ref_tab = ft['referred_table']
                cons_col = ft['constrained_columns']
                ref_col = ft['referred_columns']
                if ref_tab not in temp:
                    temp[ref_tab] = {}
                for c, r in zip(cons_col, ref_col):
                    temp[ref_tab][c] = r
            out[t] = temp
        return out

    def db_staging(self) -> None:
        """Workflow for getting the db schema information."""
        self._current_db_staging()
        self._generate_db_files(self.table_fields, self.table_foreign_keys)

    def _current_db_staging(self) -> None:
        """
        Creates DB files under ~/staging/logs. Uses _staging_file_read()
        and _staging_file_write() support for reading and writing.

        'db_tables.json' will contain the current status of the DB: 'all',
        with the last changes in 'new' and 'delete'.

        'db_tables_(date).json' will contain any changes of the DB for a
        specific 'date'.

        If db_tables.json file does not exist, it gets created with all
        tables.
        """

        d = {'all': self.db_tables}
        name = 'db_tables'
        if name in self.local.staging_path['logs']:
            db_json = self._staging_file_read('logs', name)
            a = set(db_json['all'])
            b = set(self.db_tables)

            d['delete'] = list(a.difference(b))
            d['new'] = list(b.difference(a))

            self._staging_file_write(d, 'logs', name)

            if d['delete'] or d['new']:
                now = datetime.now().strftime('%Y-%m-%d %H:%M')
                self._staging_file_write(d, 'logs', f'db_tables_{now}.json')
        else:
            d['new'] = []
            d['delete'] = []
            self._staging_file_write(d, 'logs', name)

    def _staging_file_read(self, level: str, name: str) -> json:
        """
        Parameters
        ----------
        name: str

        Returns
        -------
        out: dict()
        """

        if name in self.local.staging_path[level]:
            path = self.local.staging_path[level][name]
            with open(path, 'r') as file:
                f_json = json.load(file)
        else:
            raise FileNotFoundError
        return f_json

    def _staging_file_write(self, data: dict, level: str, name: str) -> None:
        """
        Generates
        ---------
        ~/staging/logs/{name}.json

        Parameters
        ----------
        data: dict()
        name: str
        """

        base_dir = self.local.path['staging'][level]
        fn = os.path.join(base_dir, name)
        with open(fn, 'w') as file:
            json.dump(data, file, indent=4)

    def _generate_db_files(self,
                           table_fields: dict,
                           table_foreign_keys: dict) -> None:
        """

        Parameters
        ----------
        table_fields: dict()
        table_foreign_keys: dict()

        """

        self._staging_file_write(table_fields, 'root', 'table_fields.json')
        self._staging_file_write(table_foreign_keys, 'root',
                                 'table_foreign_keys.json')


class WasteheroUpdate(WasteheroDA):
    """Class for fetching data from DB."""

    def __init__(self,
                 name: str,
                 tables: dict,
                 db_url: str,
                 geom_col: str,
                 local: interface.Interface = Interface,
                 debug: bool = False):
        super(WasteheroUpdate, self).__init__(db_url, local, debug)

        self.name = name
        self.root = tables["root"]
        self.relevant_tables = tables["tables"]
        self.categories = tables["categories"]
        self.constraints = tables["constraints"]
        self.drop = tables["drop"]
        self.exceptions = tables["exceptions"]
        self.order_by = tables["order_by"]
        self.cast = tables["cast"]
        self.geom_col = geom_col
        self.start = 0
        self.end = 0
        self.query = QueryBuilder(self)

    @cached_property
    def table_fields(self) -> json:
        """Reads db schema information from ~/staging/logs."""
        path = self.local.staging_path['root']['table_fields']
        with open(path) as f:
            tc_simple = json.load(f)
        return tc_simple

    @cached_property
    def table_foreign_keys(self) -> json:
        """Reads table fields information from ~/staging/logs."""
        path = self.local.staging_path['root']['table_foreign_keys']
        with open(path) as f:
            tf_simple = json.load(f)
        return tf_simple

    @cached_property
    def staged_table_fields(self) -> json:
        """Reads db schema information from ~/staging/edit."""
        base_dir = self.local.path['staging']['edit']
        name = self.name + '_table_fields.json'
        path = os.path.join(base_dir, name)

        with open(path) as f:
            table_fields = json.load(f)
        return table_fields

    @cached_property
    def staged_foreign_keys(self) -> json:
        """Reads db schema information from ~/staging/edit."""
        base_dir = self.local.path['staging']['edit']
        name = self.name + '_foreign_keys.json'
        path = os.path.join(base_dir, name)

        with open(path) as f:
            foreign_keys = json.load(f)
        return foreign_keys

    @cached_property
    def dataset_table_fields(self) -> dict:
        """Creates subset of table_fields according to tables."""
        out = {t: self.table_fields[t] for t in self.relevant_tables}
        return out

    @cached_property
    def dataset_foreign_keys(self) -> dict:
        """
        Creates subset of foreign_keys according to tables.
        """
        out = {t: self.table_foreign_keys[t] for t in self.relevant_tables}
        return out

    @cached_property
    def dataset_table_fields_drop(self) -> dict:
        """
        Marks dataset_table_fields as 'use': False according to
        tables, 'drop'.
        """
        table_fields_drop = self.dataset_table_fields.copy()
        for table, ds in self.drop.items():
            for drop in ds:
                table_fields_drop[table][drop]["use"] = False
        return table_fields_drop

    @cached_property
    def dataset_table_fields_no_idfk(self) -> dict:
        """Marks dataset 'id' fields and foreign keys to 'use': False."""
        false = {'out_name': None, 'use': False}
        id_re = re.compile('(^)*[_]*id$')
        tab_copy = self.dataset_table_fields_drop.copy()
        out = {t: {k: false if id_re.search(k) else v
                   for k, v in f.items()}
               for t, f in tab_copy.items()}
        return out

    @cached_property
    def dataset_table_fields_exceptions_idfk(self) -> dict:
        """
        Marks dataset_table_fields as 'use': False according to
        tables, 'drop'.
        """
        table_fields_out = self.dataset_table_fields.copy()
        for table, es in self.exceptions.items():
            for field, e in es.items():
                table_fields_out[table][field] = e
        return table_fields_out

    @cached_property
    def dataset_table_fields_exceptions(self) -> dict:
        """
        Edits dataset_table_fields according to exceptions from
        tables.
        """
        table_fields = self.dataset_table_fields_no_idfk.copy()
        for table, v in self.exceptions.items():
            for field, e in v.items():
                table_fields[table][field] = e
        return table_fields

    @cached_property
    def graph(self) -> networkx.DiGraph:
        """
        Generates a graph to understand the DB schema. Will be used for
        joins in class QueryBuilder.

        Generates graph representation of DB schema.

        Each table in 'dataset_table_fields' is considered a node.

        Each link in 'dataset_foreign_keys' and part of tables
        that is marked as 'use'.: True in dataset_table_fields_drop, is
        considered an edge.
        """

        g = nx.DiGraph()
        for table in self.dataset_table_fields.keys():
            g.add_node(table)

        for from_, v in self.dataset_foreign_keys.items():
            for to in v.keys():
                if to in self.relevant_tables:
                    constraints = self.dataset_foreign_keys[from_][to]
                    link = [self.dataset_table_fields_drop[from_][k]["use"]
                            for k in constraints.keys()]
                    if any(link):
                        g.add_edge(from_, to)
        return g

    def get_tree(self):
        t = nx.bfs_tree(self.graph, self.root)
        return t

    def draw_tree(self):
        fig, ax = plt.subplots()
        g = self.graph
        t = self.get_tree()
        _lt = list(nx.bfs_edges(g, source=self.root))
        pos = nx.planar_layout(g)
        nx.draw(t, pos=pos, ax=ax)
        nx.draw_networkx_labels(t, pos=pos, ax=ax)
        # nx.draw_networkx_edge_labels(t, pos=pos, ax=ax)
        return fig

    @cached_property
    def graph_old(self) -> networkx.DiGraph:
        """
        Generates a graph to understand the DB schema. Will be used for
        joins in class QueryBuilder.

        Generates graph representation of DB schema.

        Each table in 'dataset_table_fields' is considered a node.

        Each link in 'dataset_foreign_keys' and part of tables
        that is marked as 'use'.: True in dataset_table_fields_drop, is
        considered an edge.
        """

        g = nx.DiGraph()
        for table in self.dataset_table_fields.keys():
            g.add_node(table)

        for from_, v in self.dataset_foreign_keys.items():
            for to in v.keys():
                use_field = \
                    self.dataset_foreign_keys[from_][to][
                        "constrained_columns"][0]
                if self.dataset_table_fields_drop[from_][use_field]["use"] \
                        and to in self.relevant_tables:
                    g.add_edge(from_, to)

        return g

    @cached_property
    def bfs_search(self) -> List:
        """
        Makes a bfs search from self.graph and self.root as starting point.
        """

        g = self.get_tree()
        return list(nx.bfs_edges(g, source=self.root))

    @cached_property
    def updated_at(self) -> str:
        return datetime.today().strftime('%Y-%m-%d %H:%M')

    @cached_property
    def get_dataset_file_size(self) -> float:
        """Gets dataset file size in MegaBytes."""
        path = self.local.data_set_path['interim'][self.name]
        return os.path.getsize(path) / float(1 << 20)  # MegaBytes

    def get_data(self,
                 categorize: bool = True) -> pandas.DataFrame:
        """Workflow for fetching data from db."""
        self.start = timer()
        self._generate_dataset_files(self.dataset_table_fields_exceptions,
                                     self.dataset_foreign_keys)
        dataset = self.query_db()
        if categorize:
            dataset = self.categorise_fields(dataset)
        self.end = timer()
        if self.debug:
            self.print_debugger_info(dataset)
        return dataset

    def _generate_dataset_files(self,
                                table_fields: dict,
                                foreign_keys: dict) -> None:
        """
        Creates JSON files under /edit.

        Generates
        ---------
        ~/staging/edit/dataset_table_fields
        ~/staging/edit/dataset_foreign_keys

        Parameters
        ----------
        table_fields: dict()
        foreign_keys: dict()
        """
        base_dir = self.local.path['staging']['edit']
        name = self.name + '_table_fields.json'
        with open(os.path.join(base_dir, name), 'w') as f:
            json.dump(table_fields, f, indent=4)

        name = self.name + '_foreign_keys.json'
        with open(os.path.join(base_dir, name), 'w') as f:
            json.dump(foreign_keys, f, indent=4)

    def query_db(self) -> geopandas.GeoDataFrame:
        """
        Queries de DB and returns a geopandas GeoDataFrame.

        None values are changed as NaN.

        Returns
        -------
        out: geopandas GeoDataFrame
        """

        start_q = timer()
        self.query.build()
        end_q = timer()
        print(f'QueryBuild took :{end_q - start_q:3.2f} sec.')
        start_sql = timer()
        try:
            self.engine.metadata.reflect(self.engine)
            print(f'SQL Query Sent')
            if self.debug:
                print(str(self.query))
        except TimeoutError:
            end_sql = timer()
            raise TimeoutError(
                f'Connection pool timed out after {end_sql - start_sql} sec.')
        finally:
            results = gp.read_postgis(str(self.query),
                                      con=self.engine,
                                      geom_col=self.geom_col,
                                      crs="epsg:4326")
            end_sql = timer()
            print(f'SQL request took :{end_sql - start_sql:3.2f} sec.')
            self.engine.dispose()
        return results

    @staticmethod
    def _empty_str_to_none(s: str) -> Union[str, float]:
        if isinstance(s, str) and not s:
            return np.nan
        else:
            return str(s)

    def categorise_fields(self, dataset: pandas.DataFrame) \
            -> pandas.DataFrame:
        """
        Transforms GeoDataFrame types object to category when possible.

        Transforms str values '' to NaN with empty_str_to_none.

        Parameters
        ----------
        dataset: geopandas GeoDataFrame

        Returns
        -------
        out: geopandas GeoDataFrame
        """
        out = dataset.copy(deep=True)
        for c in self.categories:
            try:
                out[c] = out[c].apply(self._empty_str_to_none) \
                               .astype('category')
                print(f"Categorized {c}")
            except TypeError:
                print(f"Not categorized {c}")
        return out

    def save_dataset(self, dataset: Any) -> None:
        """
        Writes dataset file under ~.wh_data/data/interim.

        Raises
        ------
        ValueError when there are columns with the same name
        """

        self.local.save_dataset(dataset, 'interim', self.name)

        return

    def wh_data_logs(self, dataset: pandas.DataFrame) -> None:
        """
        Generates 'dataset_data_log.csv' under '~/staging/logs for
        keeping track with dataset information growing rate.

        Information tracking for 'mask_dataset'
        -------------------------------------
        (query_time | dataset_file_size | nº measurements | nº mask
        measurements | update time)
        """

        file_size = self.get_dataset_file_size
        base_dir = self.local.path['staging']['logs']
        name = f"{self.name}_data_log.csv"
        path = pathlib.Path(base_dir, name)

        if self.name == "mask":
            data = [[self.end - self.start,
                     file_size,
                     dataset.raw_measurements.count(),
                     dataset.fill_level_masked.count(),
                     len(dataset['container_id'].unique()),
                     len(dataset['name_project'].unique()),
                     self.updated_at]]

            columns = ["query_time",
                       "file_size_MB",
                       "measurements",
                       "mask_measurements",
                       "number_sensors",
                       "number_projects",
                       "updated_at"]

            df = pd.DataFrame(data, columns=columns, index=[0])

            if path.is_file():
                df.to_csv(path, mode='a', header=False)
            else:
                df.to_csv(path)
        else:
            print(f'WH Data logs for dataset {self.name} not implemented')

    def print_debugger_info(self, dataset: pandas.DataFrame) -> None:
        """Console print additional information."""

        if self.debug:
            print(dataset.info())
            print(f"Data fetching time elapsed: {self.end - self.start}")
            print(f"Updated at: {self.updated_at}")
            print(f"Weight: {self.get_dataset_file_size} MB")


# def last_update_info(data_set: str) -> str:
#     """
#     Displays when data were last update according to dataset if exists.
#     """
#
#     name = f"{data_set}_dataset"
#     dataset = os.path.join(PATHS.INTERMEDIATE_DATA_PATH, name)
#     if os.path.isfile(dataset):
#         stat_result = os.stat(dataset)
#         timestamp_str = datetime.fromtimestamp(stat_result.st_mtime). \
#             strftime('%Y-%m-%d %H:%M')
#         return f"Last data update: {timestamp_str}"
#     else:
#         return ""


def get_available_tables(root_dir: pathlib.Path) -> json:
    package_dir = root_dir
    path = package_dir.joinpath('relevant_tables.json')
    if not path.exists():
        raise ValueError(f'relevant_tables.json not found in {package_dir}')
    with path.open() as f:
        tables = json.load(f)

    return tables


def get_relevant_tables(table_set: str) -> json:
    """
    Reads relevant_tables from ~/. according to 'table_set'.

    Parameters
    ----------
    table_set: str() [mask | dashboard] expected

    Returns
    -------
    dict()

    Raises
    ------
    ValueError when unexpected argument
    """

    tables = get_available_tables(root_dir=root_dir)

    if table_set in tables:
        return tables[table_set]
    else:
        raise ValueError(f"Table set {table_set} not found in "
                         f"relevant_tables.json")


# def get_data_load(data: pandas.DataFrame,
#                   data_set: str = 'mask'):
#     if data_set == 'mask':
#         db = SensorDB(db=data)
#         data_load = db.dataset
#     elif data_set in ['weight', 'weight_historical', 'weight_total']:
#         fields = {'company': 'name_company',
#                   'project': 'name_project',
#                   'container': 'container_id',
#                   'fraction': 'waste_fraction_id'}
#         sdb = SensorDB(**Interface.get_dataset('interim', 'mask'),
#                        fields=fields)
#         db = WeightSensorDB(db=data, db_s=sdb, fields=fields)
#         data_load = db.dataset
#     else:
#         data_load = {'db': data}
#     return data_load


# def merge_weight() -> None:
#     weight_fields = {'company': 'name_company',
#                      'project': 'name_project',
#                      'container': 'container_id',
#                      'fraction': 'waste_fraction_id'}
#
#     sensor_fields = {'company': 'name_company',
#                      'project': 'name_project',
#                      'container': 'container_id',
#                      'device': 'container_device_id'}
#     wh_sensor = Interface.get_dataset('interim', 'mask')
#     wh_weight = Interface.get_dataset('interim', 'weight')
#     wh_w_hist = Interface.get_dataset('interim', 'weight_historical')
#     SDB = SensorDB(db=wh_sensor['db'],
#                    fields=sensor_fields,
#                    time='created_at')
#     WHDB = WeightSensorDB(db=wh_w_hist['db'],
#                           time='completed_at',
#                           fields=weight_fields,
#                           db_s=SDB,
#                           paired_data=wh_w_hist['paired_data'])
#     WDB = WeightSensorDB(db=wh_weight['db'],
#                          time='completed_at',
#                          fields=weight_fields,
#                          db_s=SDB,
#                          paired_data=wh_weight['paired_data'])
#
#     print('Merging Weight Data')
#     if WDB is None:
#         print('WDB not present: skipping Merge')
#     elif WHDB is None:
#         print('WHDB not present: skipping Merge')
#     else:
#         merged = WDB + WHDB
#         Interface.save_dataset(merged.dataset, 'interim', 'weight_total')
#     return


# def update_data(data_set: str = 'mask',
#                 debug: bool = False,
#                 do_staging: bool = False,
#                 do_merge: bool = False) -> None:
#     if do_staging:
#         start_s = timer()
#         whst = WasteheroStaging(config('DB_URL'))
#         whst.db_staging()
#         end_s = timer()
#         print(f'Staging took :{end_s - start_s:3.2f} sec.')
#
#     current_data = Interface.get_dataset('interim', data_set)['db']
#
#     tabs = get_relevant_tables(data_set)
#
#     # Identify the field used to order the data-set
#     # This field is also used to filter new-data
#     t_ord = list(tabs['order_by'].keys())[0]
#     f_ord = list(tabs['order_by'][t_ord].keys())[0]
#     f_ord_df = f_ord
#
#     # check if f_ord name has been changed
#     if t_ord in tabs['exceptions']:
#         if f_ord in tabs['exceptions'][t_ord]:
#             f_ord_df = tabs['exceptions'][t_ord][f_ord]['out_name']
#
#     start_from = current_data[f_ord_df].sort_values().iloc[-1].isoformat()
#     tabs['constraints'][t_ord][f_ord] = ['>', f'\'{start_from}\'']
#     whup = WasteheroUpdate(name=data_set,
#                            tables=tabs,
#                            geom_col='coordinates',
#                            db_url=config('DB_URL'),
#                            debug=debug)
#     # REMEMBER TO CATEGORIZE AFTER APPENDING AND BEFORE SAVING!
#     new_data = whup.get_data(categorize=False)
#     new_rows = len(new_data)
#     print(f'New Rows added: {new_rows}')
#     if new_rows == 0:
#         return
#     data = current_data.append(new_data, ignore_index=True)
#     # CATEGORIZING !!
#     data = whup.categorise_fields(data)
#     data_load = get_data_load(data=data, data_set=data_set)
#     whup.save_dataset(data_load)
#     whup.wh_data_logs(data)
#     if (data_set in ['weight', 'weight_historical']) \
#             and do_merge \
#             and (new_rows > 0):
#         merge_weight()
#     del current_data
#     del new_data
#     return
#
#
# def fetch_data(data_set: str = 'mask',
#                debug: bool = False,
#                do_staging: bool = True) -> None:
#     """
#     Marked as entry point in setup, used in api.
#
#     """
#     if do_staging:
#         start_s = timer()
#         whst = WasteheroStaging(config('DB_URL'), debug=debug)
#         whst.db_staging()
#         end_s = timer()
#         print(f'Staging took :{end_s - start_s:3.2f} sec.')
#     tabs = get_relevant_tables(data_set)
#     whup = WasteheroUpdate(name=data_set,
#                            tables=tabs,
#                            geom_col='coordinates',
#                            db_url=config('DB_URL'),
#                            debug=debug)
#     data = whup.get_data(categorize=True)
#     data_load = get_data_load(data=data, data_set=data_set)
#     whup.save_dataset(data_load)
#     whup.wh_data_logs(data)
#     return
#
#
# def update_all_data(ds_list: Tuple[str] = ('mask',
#                                            'weight',
#                                            'weight_historical'),
#                     debug: bool = False,
#                     do_staging: bool = False,
#                     do_merge: bool = False) -> None:
#     if do_staging:
#         start_s = timer()
#         whst = WasteheroStaging(config('DB_URL'))
#         whst.db_staging()
#         end_s = timer()
#         print(f'Staging took :{end_s - start_s:3.2f} sec.')
#
#     for ds in ds_list:
#         print(f'Update {ds} data-set')
#         update_data(ds, debug=debug, do_staging=False)
#     if any_in(ds_list, ['weight', 'weight_historical']) and do_merge:
#         print(f'Merging weight data-set')
#         merge_weight()
#     pathlib.Path(PATHS['DATA_ROOT']).touch()
#
#
# def init_all_data() -> None:
#     # Adding argument parser to accept the force flag
#     parser = ArgumentParser(description="Initialize all data")
#     parser.add_argument("--force",
#                         help="force dataset download",
#                         action="store_true")
#     args = parser.parse_args()
#
#     # Weight Data partially depends on the "Mask" data-set, so the
#     # Order of the ds_list should not be changed !!!
#     ds_list = ('mask', 'weight', 'weight_historical')
#     s = True
#     merge = False
#     for ds in ds_list:
#         # Check if dataset exists
#         if ds in Interface.data_set_path["interim"] and not args.force:
#             print(f"{ds} dataset is already available, skipping. Use --force to force download.")
#         else:
#             fetch_data(data_set=ds, do_staging=s)
#             if ds in ['weight', 'weight_historical']:
#                 merge = True
#             s = False
#     if merge:
#         merge_weight()
#     pathlib.Path(PATHS['DATA_ROOT']).touch()
#     return
#
#
# def fetch_all_data(ds_list: Tuple[str] = ('mask', 'weight',
#                                           'weight_historical'),
#                    debug=False,
#                    do_staging=True,
#                    do_merge: bool = False) -> None:
#     # Weight Data partially depends on the "Mask" data-set, so the
#     # Order of the ds_list should not be changed !!!
#     if do_staging:
#         start_s = timer()
#         whst = WasteheroStaging(config('DB_URL'))
#         whst.db_staging()
#         end_s = timer()
#         print(f'Staging took :{end_s - start_s:3.2f} sec.')
#     for ds in ds_list:
#         print(f'Fetching {ds} data-set:')
#         fetch_data(ds, debug=debug, do_staging=False)
#     if any_in(ds_list, ['weight', 'weight_historical']) and do_merge:
#         merge_weight()
#     pathlib.Path(PATHS['DATA_ROOT']).touch()
#     return


def re_stage():
    start_s = timer()
    whst = WasteheroStaging(config('DB_URL'))
    whst.db_staging()
    end_s = timer()
    print(f'Staging took :{end_s - start_s:3.2f} sec.')

#
# def run() -> None:
#     """Entry point."""
#
#     update_choices = list(get_available_tables()) + ['all']
#
#     parser = ArgumentParser(description="WasteHero Data Pipeline Interface")
#
#     parser.add_argument("-d", "--dataset",
#                         help="Fetches data related to functionality",
#                         action="store",
#                         choices=update_choices,
#                         default='all')
#
#     parser.add_argument("-u", "--update",
#                         help="Fetches only new data",
#                         action="store_true")
#
#     parser.add_argument("-s", "--staging",
#                         help="Review and validation of the DB",
#                         action="store_true")
#
#     parser.add_argument("--debug",
#                         help="activates debug",
#                         action="store_true")
#
#     parser.add_argument("--no_merge",
#                         help="do not merge Weight DBs",
#                         action="store_false")
#
#     args = parser.parse_args()
#
#     if args.dataset == "all":
#         if args.update:
#             update_all_data(debug=args.debug,
#                             do_staging=args.staging,
#                             do_merge=args.no_merge)
#         else:
#             fetch_all_data(debug=args.debug,
#                            do_staging=args.staging,
#                            do_merge=args.no_merge)
#     else:
#         if args.update:
#             update_data(data_set=args.dataset,
#                         debug=args.debug,
#                         do_staging=args.staging,
#                         do_merge=args.no_merge)
#         else:
#             fetch_data(data_set=args.dataset,
#                        debug=args.debug,
#                        do_staging=args.staging)
