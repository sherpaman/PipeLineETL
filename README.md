# Template for Data-Pipeline

    
## Setup 

Note: it is highly suggested to create a virtual environment

```
$ python -m virtualenv <venv_name>
```

To activate the virtual env:
```
$ source <venv_name>/bin/activate
```
To de-activate the virtual env:
```
$ deactivate
```

# Dataset cookbook

The file `relevant_tables.json` is used for specifying datasets information regarding how to build an SQL query for it and what tables compose it.

The file is `json` valid and formatted as follows:

| Key | Description |
|-----|-------------|
| Dataset | array of `dataset` objects describing its information |

### Dataset

| Dataset | Description |
|-----|-------------|
| `root` | `str` that indicates the root of a set of tables |
| `tables` | array of `str` in which to fetch data from for the `dataset` |
| `drop` | array of `key, list()` objects describing, from what table, what fields to not fetch **(Before SQL query)** |
| `exceptions` | array of `key, dict()` object escribing, from what table, what fields to KEEP or RENAME **(Before SQL query)**|
| `constraints` | array of `key, dict()` describing, to what table and fields, add an SQL-constraint **(SQL query)** |
| `order_by` | array of `key, dict()` describing, to what table and fields, add SQL-sorting options |
| `cast` | array of `key, dict()` describing, to what table and fields, apply an SQL-cast **(SQL query)** |
| `categories` | `list()` with column names to convert to 'category' in a Pandas dataframe **(After SQL query)** |


### Dataset cookbook example*

```
{
    "mask": {
        "root": "analytics_filllevelmeasurement",
        "tables": [
            "analytics_filllevelmeasurement",
            "device_devicetocontainer",
            "device_container",
        ],
        "drop": {
            "analytics_filllevelmeasurement": [
                "gwt",
                "rsi",
                "snr"
            ],
            "device_container": [
                "photo",
                "meta_data"
            ],
        },
        "exceptions": {
            "analytics_filllevelmeasurement": {
                "container_device_id": {
                    "out_name": null,
                    "use": true
                }
            },
            "device_devicetocontainer": {
                "container_id": {
                    "out_name": "container_name_list",
                    "use": true
                }
            }

        },
        "constraints": {
            "analytics_filllevelmeasurement": {
                "raw_measurements": "is not null",
                "created_at": "> '2021-03-03'"
            }
        },
        "order_by": {
            "analytics_filllevelmeasurement": {
                "created_at": "asc"
            }
        },
        "cast": {
            "point": [
                {
                    "cast": "ST_X",
                    "out_name": "longitude"
                },
                {
                    "cast": "ST_Y",
                    "out_name": "latitude"
                }
            ]
        },
        "categories": [
            "name",
        ]
    }...
```

*In order to apply any changes, it is taken for granted an advanced knowledge of WasteHero database: schema tables and fields.
