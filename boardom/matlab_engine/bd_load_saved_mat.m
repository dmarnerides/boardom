function res = bd_load_saved_mat(full_path, var_name)
    res = getfield(load(full_path, '-mat', var_name), var_name);
end
