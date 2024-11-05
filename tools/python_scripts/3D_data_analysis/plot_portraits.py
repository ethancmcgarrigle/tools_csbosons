import plot_density_portrait
import plot_spins_portrait

def main():
  # wrapper program to plot density profiles by chunk averaging
  # chunk average len

  # for each chunk of samples 
  # 1. Split the samples 
  # 2. Average the samples 
  # 3. run dat2vtk on the sample

  N_samps_to_avg = 4
  max_samps = 10     #hardcoded for now
  #fig_storage_prefix = "/home/emcgarrigle/Junheng_collab/Na23/3D_pop_imbalance_Tsweep/B_0.1_density_profiles/"
  fig_storage_prefix = "/home/emcgarrigle/Junheng_collab/Na23/3D_pop_imbalance_Tsweep/B_0.1_mag_profiles/"

  for start_index in range(5, max_samps, N_samps_to_avg):
    # generates sequence starting at 1, going to max_samps, in increments in N_samps_to_avg -- use this as start index
    # in the processing script
    print('Averaging starting at ' + str(start_index))
    #plot_density_portrait.main(N_samps_to_avg, start_index, fig_storage_prefix)
    plot_spins_portrait.main(N_samps_to_avg, start_index, fig_storage_prefix)

if __name__ == '__main__':
  main()
