echo "----single human pose experiment in 2026----"
python /home/tonic/guan125/3DF-CSI-HPE/Arranged/heatmap.py \
    --data_path /home/tonic/CSI-dataset/20260320 \
    --save_path /home/tonic/guan125/1exp_data/20260320 \
    --exp_name 20260320-right-hand-fb \
    --heatmap_type ToF-Doppler \
    --save_fig \
    --save_mat \
    # --not_create_new_steering_matrix


# available AoA-ToF-Doppler ToF-Doppler