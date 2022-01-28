from functools import cached_property

from sqlalchemy import Table
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..tools.utils import is_iterable


class QueryBuilder:
    def __init__(self, whup):
        self.whup = whup
        self.engine = whup.engine
        self.select = None

    def __str__(self):
        if self.select is None:
            return 'Query has not been built yet'
        else:
            return str(self.select.statement.compile(
                compile_kwargs={"literal_binds": True}))

    @cached_property
    def cast(self):
        # This list has to be manually maintained up-to-date
        out = {'ST_X': func.ST_X,
               'ST_Y': func.ST_Y
               }
        return out

    @staticmethod
    def do_filter(i, *args):
        # This is gate-keeping function
        c = args
        if c[0] == "is not null":
            out = i.isnot(None)
        elif c[0] == "is null":
            out = i.is_(None)
        elif c[0] == "is":
            out = i.is_(c[1])
        elif c[0] == ">":
            out = i > c[1]
        elif c[0] == "<":
            out = i < c[1]
        elif c[0] == ">=":
            out = i >= c[1]
        elif c[0] == "<=":
            out = i <= c[1]
        elif c[0] == "==":
            out = i == c[1]
        else:
            return None
        print(f'add filter {i} {c}')
        return out

    @property
    def session(self):
        return Session(self.engine)

    @cached_property
    def tables(self):
        out = {t: Table(t, self.engine.metadata, autoload_with=self.engine)
               for t in self.whup.relevant_tables}
        return out

    @cached_property
    def columns(self):
        columns = []
        # staged_table_fields:
        # t : table, str
        # f : field, dict { 'key' : { option_values } }
        # k : key
        # v : option_values, dict { 'out_name': str, 'use': bool }
        for t, f in self.whup.dataset_table_fields_exceptions.items():
            tab = self.tables[t]
            for k, v in f.items():
                if v["use"]:
                    # cast, dict { k : [ list(casts) ] }
                    # casts, dict {'cast' : sql_function, str,
                    #              'out_name' : str }
                    if k in self.whup.cast.keys():
                        for c_k, opt in self.whup.cast.items():
                            for o in self.whup.cast[k]:
                                c_f = self.cast[o["cast"]]
                                as_name = o["out_name"]
                                columns.append(c_f(tab.c[k]).label(as_name))
                    else:
                        as_name = v["out_name"]
                        if as_name is None:
                            columns.append(tab.c[k])
                        else:
                            columns.append(tab.c[k].label(as_name))
        return columns

    def init_select(self):
        self.select = self.session.query(*self.columns)
        return None

    def get_select(self, refresh=False):
        if (self.select is None) or refresh:
            self.init_select()
            return self.select
        else:
            return self.select

    def join(self):
        stm = self.get_select()
        for f, t in self.whup.bfs_search:
            to = self.tables[t]
            from_ = self.tables[f]
            links = self.whup.dataset_foreign_keys[f][t]
            for f_k, t_k in links.items():
                if self.whup.dataset_table_fields_exceptions_idfk[f][f_k]['use']:
                    stm = stm.join(self.tables[t], from_.c[f_k] == to.c[t_k])
        self.select = stm
        return

    def join_old(self):
        stm = self.get_select()
        for f, t in self.whup.bfs_search:
            link = self.whup.dataset_foreign_keys[f][t]
            f_k = link["constrained_columns"][0]
            t_k = link["referred_columns"][0]
            to = self.tables[t]
            from_ = self.tables[f]
            stm = stm.join(self.tables[t], from_.c[f_k] == to.c[t_k])
        self.select = stm
        return

    def where(self):
        stm = self.get_select()
        if self.whup.constraints is not None:
            for k, v in self.whup.constraints.items():
                for f, c in v.items():
                    if is_iterable(c):
                        flt = self.do_filter(self.tables[k].c[f], *c)
                    else:
                        flt = self.do_filter(self.tables[k].c[f], c)
                    if flt is not None:
                        stm = stm.filter(flt)
        self.select = stm
        return

    def order_by(self):
        stm = self.get_select()
        if self.whup.order_by is not None:
            for k, v in self.whup.order_by.items():
                for f, o in v.items():
                    if o == "asc":
                        stm = stm.order_by(self.tables[k].c[f])
                    elif o == "desc":
                        stm = stm.order_by(self.tables[k].c[f].desc())
                    else:
                        # ignore anything else
                        continue
        self.select = stm
        return

    def build(self):
        self.init_select()
        self.join()
        self.where()
        self.order_by()
        return
