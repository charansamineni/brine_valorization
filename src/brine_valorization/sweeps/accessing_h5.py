import h5py

h5_file = "voltageub1200_recovery_sweep_70gL_08-06_0930.h5"

def get_values(f, variable_path):
    return f[variable_path + "/value"][:]

with h5py.File(h5_file, "r") as f:

    recovery = get_values(f, "sweep_params/NaCl Recovery")
    # lcop = get_values(f, "outputs/fs.costing.LCOP")
    # hcl = get_values(f, "outputs/fs.bpmed.flow_mass_product[bpmed,HCl]")
    # naoh = get_values(f, "outputs/fs.bpmed.flow_mass_product[bpmed,NaOH]")
    # voltage = get_values(f, "outputs/fs.bpmed.bpmed[0].voltage_applied[0.0]")
    # opex = get_values(f, "outputs/fs.costing.total_operating_cost")
    # capex = get_values(f, "outputs/fs.costing.total_capital_cost")
    splitting_flux_end = get_values(f, "outputs/fs.bpmed.bpmed[0].flux_splitting[0.0,1.0]")
    splitting_flux_middle = get_values(f, "outputs/fs.bpmed.bpmed[0].flux_splitting[0.0,0.5]")
    splitting_flux_start = get_values(f, "outputs/fs.bpmed.bpmed[0].flux_splitting[0.0,0.0]")
    hcl_concentration = get_values(f, "outputs/fs.bpmed.product_mass_concentration[bpmed,HCl]")
    naoh_concentration = get_values(f, "outputs/fs.bpmed.product_mass_concentration[bpmed,NaOH]")
    lim_current_dens_inlet = get_values(f, "outputs/fs.bpmed.bpmed[0].current_dens_lim_bpem[0.0,0.0]")
    lim_current_dens_middle = get_values(f, "outputs/fs.bpmed.bpmed[0].current_dens_lim_bpem[0.0,0.5]")
    lim_current_dens_end = get_values(f, "outputs/fs.bpmed.bpmed[0].current_dens_lim_bpem[0.0,1.0]")



    # for r, l, h, n, v, o, c in zip(recovery, lcop, hcl, naoh, voltage, opex, capex):
    #     print(r, l, h, n, v, o, c)

    for r, li, lm, le in zip(recovery, lim_current_dens_inlet, lim_current_dens_middle, lim_current_dens_end):
        print(r, li, lm, le)