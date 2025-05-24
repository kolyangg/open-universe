#!/usr/bin/env bash
set -e

# Example calls (replace with your own links and destination folders):
echo "Downloading checkpoint files..."

echo "Downloading Univ++ original checkpoint (50k steps)"
python3 models/universe/utils/checkp_dl.py "https://drive.google.com/file/d/1l2cDMbI-NDV2LV4tcjfyr0YpideGTzjx/view?usp=sharing" "exp2/orig_bucket_cluster"

echo "Downloading Miipher+Double checkpoint"
python3 models/universe/utils/checkp_dl.py  "https://drive.google.com/file/d/13nBAsJfrOc4UVdKQSzx4HbtDhqtL6kUR/view?usp=sharing" "exp2/wv_double+ff_m2_wvloss_18May"

echo "Downloading Simple+Large checkpoint"
python3 models/universe/utils/checkp_dl.py  "https://drive.google.com/file/d/1pD0VfkfGeMt_jKoTLGOJEfFtmkXZn7LT/view?usp=sharing" "exp2/16May_full_film_xph_wv"

