import argparse
import json
import numpy as np
import os
import gc
from pathlib import Path
from scipy.io import loadmat,savemat
from utilsforheatmap import (
	heatmap_setup,
	CSI_preprocessing,
	create_steering_matrix_ToF_Doppler,
	create_steering_matrix_F3D,
	smoothed_CSI,
	calculate_correlation_matrix,
	pipeline_3D,
	run_music_algorithm,
)
from plot_utils import (
	generate_heatmaps,	
)

def heatmap_gen(args):
	# load the config file of the experiment
	config_path = Path(__file__).resolve().parent/ "config.json"
	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)

	for exp_name in args.exp_names:

		plot_gt = False
		# load row CSI data
		print(f'Experiment: {exp_name}')
		CSI = np.load(os.path.join(args.data_path,'csi', f"csi_{exp_name}.npz"))['csi']
		
		# correct the antenna order original
		CSI = CSI[:, :, config["antenna_order"], :]
		
		# prepare heatmap parameters
		heatmap_setting = heatmap_setup(config)

		# Data preprocessing
		CSI_mov = CSI_preprocessing(config, CSI, heatmap_setting)

		# free memory
		del CSI
		gc.collect()

		# Heatmap pipeline
		for heatmap_type in args.heatmap_type:

			print(f"Starting {heatmap_type} method:")
			
			if "ToF-Doppler" == heatmap_type:
				# create steering matrix for AoA-ToF heatmap
				steering_matrix_ToF_Doppler = create_steering_matrix_ToF_Doppler(heatmap_setting)

				# get the smoothed csi
				# output:(timestamp, tx, steps_smooth_AoA, steps_smooth_ToF, window_rx, window_subcarrier)
				CSI_smoothed = smoothed_CSI(heatmap_type,heatmap_setting, CSI_mov)

				# get the correlation matrix R
				# output:(timestamp, N*M, N*M)
				print("Calculating correlation matirx")
				R = calculate_correlation_matrix(CSI_smoothed,heatmap_type=heatmap_type)
				del CSI_smoothed
				gc.collect()

				# get spectrums
				spectrums = run_music_algorithm(R, steering_matrix_ToF_Doppler)
				del R, steering_matrix_ToF_Doppler
				gc.collect()

				# save as figure and mat
				if args.save_mat:
					mat_path = Path(args.save_path)/"heatmap_result"/"mat"/exp_name/heatmap_type
					if not os.path.exists(mat_path):
						mat_path.mkdir(parents=True, exist_ok=True)
					output_file = os.path.join(mat_path,"smoothed_CSI_avg.mat")
					idx = np.arange(spectrums.shape[0],dtype=np.int64)
					spectrums = spectrums.astype(np.float64)
					savemat(output_file, {
						"idx":idx,
						"spectrum":spectrums
					})
				if args.save_fig:
					generate_heatmaps(exp_name, heatmap_type, spectrum=spectrums, heatmap_setting=heatmap_setting, output_folder=args.save_path,plot_gt=plot_gt)
				del spectrums
				gc.collect()

			elif "AoA-ToF-Doppler" == heatmap_type:
				# create steering matrix for AoA-Doppler heatmap
				steering_matrix_3D = create_steering_matrix_F3D(heatmap_setting)

				# get the smoothed csi
				# output:(timestamp, tx, steps_smooth_AoA, steps_smooth_ToF, window_rx, window_subcarriers, window_sample)
				CSI_smoothed = smoothed_CSI(heatmap_type,heatmap_setting, CSI_mov)

				# get spectrums and save as figure and mat
				pipeline_3D(exp_name, args, CSI_smoothed, steering_matrix_3D, heatmap_setting, start_sample=None, end_sample=None, plot_gt=plot_gt )

if __name__ == '__main__':
	# Parser	
	parser = argparse.ArgumentParser()
	
	# data config
	parser.add_argument('--data_path', default = '/home/tonic/guan125/1exp_data/20250512')
	parser.add_argument('--save_path', default = '/home/tonic/guan125/1exp_data/20250512')
	parser.add_argument('--exp_names', nargs = '+', default = ['csi_20250512-tof-2-2'])
	# plot type setting
	parser.add_argument('--heatmap_type', nargs= '+', default=['ToF-Doppler']) # available: 'ToF-Doppler' 'AoA-ToF-Doppler'
	parser.add_argument('--save_fig', dest = 'save_fig', action = 'store_true')
	parser.set_defaults(save_fig = False)
	parser.add_argument('--save_mat', dest = 'save_mat', action = 'store_true')
	parser.set_defaults(save_mat = False)

	# parse parser
	args = parser.parse_args()

	# main function
	heatmap_gen(args = args)