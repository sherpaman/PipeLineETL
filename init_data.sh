#!/bin/sh

echo "Initializing data"
start=`date +%s`
init_all
update_sensor_mask
update_fill_level_model
update_weight_model
update_weight_factor_model
end=`date +%s`
echo "Finished in $((end-start))s"