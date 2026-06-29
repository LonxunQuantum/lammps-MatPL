/*
    Copyright (c) 2026 Beijing Lonxun Quantum Co., Ltd.
    This file is distributed under the terms of the GNU General Public License
    as published by the Free Software Foundation, either version 2 of the License,
    or (at your option) any later version.

    Open-source runtime loader for libnep_gpu.so.
    Uses dlopen/dlsym to dynamically load the closed-source NEP GPU library.
    Provides a C++ wrapper class (NepModelLoader) for the NepModel engine
    so that PairNEPKokkos requires minimal modifications.

    If libnep_gpu.so is not found, the loader reports a clear error message
    and the pair style will fail gracefully at initialization time.
*/

#ifndef NEP_GPU_LOADER_H
#define NEP_GPU_LOADER_H

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>

/* ─── C ABI Function Pointer Types ────────────────────────────────────── */

extern "C" {

typedef struct nep_handle_t nep_handle_t;

typedef nep_handle_t* (*nep_create_t)(void);
typedef void         (*nep_destroy_t)(nep_handle_t*);
typedef const char*  (*nep_get_error_t)(nep_handle_t*);

typedef int  (*nep_license_load_t)(nep_handle_t*, const char*);
typedef char* (*nep_license_info_t)(nep_handle_t*);

typedef int (*nep_read_potential_t)(nep_handle_t*, const char*,
                                     int, int, int, int);
typedef int (*nep_set_atom_type_map_t)(nep_handle_t*, int, const int*);

typedef int (*nep_compute_t)(nep_handle_t*,
                              int, int, int, int, int, int,
                              int, int, int, int, int,
                              int*, int*, int*, int*,
                              double*,
                              double*, double*,
                              double*, double*,
                              double*, double*,
                              double*, int);

typedef int (*nep_reset_data_t)(nep_handle_t*, int, int, int, int);
typedef int (*nep_free_data_t)(nep_handle_t*);
typedef int (*nep_get_gpu_memory_stats_t)(nep_handle_t*, size_t*, size_t*, size_t*);
typedef int (*nep_calculate_max_atoms_t)(nep_handle_t*, int, int, int, int, int, int);

typedef double     (*nep_get_rc_radial_t)(nep_handle_t*);
typedef int        (*nep_get_n_max_radial_t)(nep_handle_t*);
typedef int        (*nep_get_n_max_angular_t)(nep_handle_t*);
typedef int        (*nep_get_mn_radial_t)(nep_handle_t*);
typedef int        (*nep_get_mn_angular_t)(nep_handle_t*);
typedef int        (*nep_get_ann_dim_t)(nep_handle_t*);
typedef int        (*nep_get_num_types_t)(nep_handle_t*);
typedef int        (*nep_get_zbl_enabled_t)(nep_handle_t*);
typedef double     (*nep_get_zbl_rc_outer_t)(nep_handle_t*);
typedef const int* (*nep_get_element_list_t)(nep_handle_t*, int*);

} // extern "C"

/* ─── NepModelLoader — dlopen-based C++ wrapper ─────────────────────────── */

class NepModelLoader {
public:
    NepModelLoader() : lib_handle_(nullptr), nep_handle_(nullptr) {}

    explicit NepModelLoader(const char* lib_path)
        : lib_handle_(nullptr), nep_handle_(nullptr)
    {
        load_library(lib_path);
    }

    ~NepModelLoader()
    {
        if (nep_handle_ && fn_destroy_) fn_destroy_(nep_handle_);
        if (lib_handle_) dlclose(lib_handle_);
    }

    // Non-copyable, movable
    NepModelLoader(const NepModelLoader&) = delete;
    NepModelLoader& operator=(const NepModelLoader&) = delete;
    NepModelLoader(NepModelLoader&& other) noexcept
        : lib_handle_(other.lib_handle_)
        , nep_handle_(other.nep_handle_)
        , fn_create_(other.fn_create_)
        , fn_destroy_(other.fn_destroy_)
        , fn_get_error_(other.fn_get_error_)
        , fn_license_load_(other.fn_license_load_)
        , fn_license_info_(other.fn_license_info_)
        , fn_read_potential_(other.fn_read_potential_)
        , fn_set_atom_type_map_(other.fn_set_atom_type_map_)
        , fn_compute_(other.fn_compute_)
        , fn_reset_data_(other.fn_reset_data_)
        , fn_free_data_(other.fn_free_data_)
        , fn_get_gpu_memory_stats_(other.fn_get_gpu_memory_stats_)
        , fn_calculate_max_atoms_(other.fn_calculate_max_atoms_)
        , fn_get_rc_radial_(other.fn_get_rc_radial_)
        , fn_get_n_max_radial_(other.fn_get_n_max_radial_)
        , fn_get_n_max_angular_(other.fn_get_n_max_angular_)
        , fn_get_mn_radial_(other.fn_get_mn_radial_)
        , fn_get_mn_angular_(other.fn_get_mn_angular_)
        , fn_get_ann_dim_(other.fn_get_ann_dim_)
        , fn_get_num_types_(other.fn_get_num_types_)
        , fn_get_zbl_enabled_(other.fn_get_zbl_enabled_)
        , fn_get_zbl_rc_outer_(other.fn_get_zbl_rc_outer_)
        , fn_get_element_list_(other.fn_get_element_list_)
        , error_msg_(std::move(other.error_msg_))
        , loaded_(other.loaded_)
        , rc_radial_(other.rc_radial_)
        , zbl_enabled_(other.zbl_enabled_)
        , zbl_rc_outer_(other.zbl_rc_outer_)
        , element_list_(std::move(other.element_list_))
    {
        other.lib_handle_ = nullptr;
        other.nep_handle_ = nullptr;
        other.loaded_ = false;
    }

    NepModelLoader& operator=(NepModelLoader&& other) noexcept
    {
        if (this != &other) {
            if (nep_handle_ && fn_destroy_) fn_destroy_(nep_handle_);
            if (lib_handle_) dlclose(lib_handle_);
            lib_handle_ = other.lib_handle_;
            nep_handle_ = other.nep_handle_;
            // copy all fn pointers...
            fn_create_ = other.fn_create_;
            fn_destroy_ = other.fn_destroy_;
            fn_get_error_ = other.fn_get_error_;
            fn_license_load_ = other.fn_license_load_;
            fn_license_info_ = other.fn_license_info_;
            fn_read_potential_ = other.fn_read_potential_;
            fn_set_atom_type_map_ = other.fn_set_atom_type_map_;
            fn_compute_ = other.fn_compute_;
            fn_reset_data_ = other.fn_reset_data_;
            fn_free_data_ = other.fn_free_data_;
            fn_get_gpu_memory_stats_ = other.fn_get_gpu_memory_stats_;
            fn_calculate_max_atoms_ = other.fn_calculate_max_atoms_;
            fn_get_rc_radial_ = other.fn_get_rc_radial_;
            fn_get_n_max_radial_ = other.fn_get_n_max_radial_;
            fn_get_n_max_angular_ = other.fn_get_n_max_angular_;
            fn_get_mn_radial_ = other.fn_get_mn_radial_;
            fn_get_mn_angular_ = other.fn_get_mn_angular_;
            fn_get_ann_dim_ = other.fn_get_ann_dim_;
            fn_get_num_types_ = other.fn_get_num_types_;
            fn_get_zbl_enabled_ = other.fn_get_zbl_enabled_;
            fn_get_zbl_rc_outer_ = other.fn_get_zbl_rc_outer_;
            fn_get_element_list_ = other.fn_get_element_list_;
            error_msg_ = std::move(other.error_msg_);
            loaded_ = other.loaded_;
            rc_radial_ = other.rc_radial_;
            zbl_enabled_ = other.zbl_enabled_;
            zbl_rc_outer_ = other.zbl_rc_outer_;
            element_list_ = std::move(other.element_list_);
            other.lib_handle_ = nullptr;
            other.nep_handle_ = nullptr;
            other.loaded_ = false;
        }
        return *this;
    }

    bool is_loaded() const { return loaded_; }
    const std::string& error() const { return error_msg_; }

    /* ── Load library and resolve all symbols ─────────────────────────── */

    bool load_library(const char* lib_path = "libnep_gpu.so")
    {
        lib_handle_ = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
        if (!lib_handle_) {
            error_msg_ = std::string("Failed to load ") + lib_path
                       + ": " + dlerror();
            loaded_ = false;
            return false;
        }

        #define LOAD_SYM(name) \
            fn_##name##_ = (nep_##name##_t)dlsym(lib_handle_, "nep_" #name); \
            if (!fn_##name##_) { \
                error_msg_ = std::string("Symbol not found: nep_") + #name; \
                dlclose(lib_handle_); \
                lib_handle_ = nullptr; \
                loaded_ = false; \
                return false; \
            }

        LOAD_SYM(create);
        LOAD_SYM(destroy);
        LOAD_SYM(get_error);
        LOAD_SYM(license_load);
        LOAD_SYM(license_info);
        LOAD_SYM(read_potential);
        LOAD_SYM(set_atom_type_map);
        LOAD_SYM(compute);
        LOAD_SYM(reset_data);
        LOAD_SYM(free_data);
        LOAD_SYM(get_gpu_memory_stats);
        LOAD_SYM(calculate_max_atoms);
        LOAD_SYM(get_rc_radial);
        LOAD_SYM(get_n_max_radial);
        LOAD_SYM(get_n_max_angular);
        LOAD_SYM(get_mn_radial);
        LOAD_SYM(get_mn_angular);
        LOAD_SYM(get_ann_dim);
        LOAD_SYM(get_num_types);
        LOAD_SYM(get_zbl_enabled);
        LOAD_SYM(get_zbl_rc_outer);
        LOAD_SYM(get_element_list);
        #undef LOAD_SYM

        nep_handle_ = fn_create_();
        if (!nep_handle_) {
            error_msg_ = "nep_create() returned NULL";
            dlclose(lib_handle_);
            lib_handle_ = nullptr;
            loaded_ = false;
            return false;
        }

        loaded_ = true;
        return true;
    }

    /* ── License ──────────────────────────────────────────────────────── */

    int license_load(const char* path) {
        return fn_license_load_(nep_handle_, path);
    }

    /** Return the last error message from the C library (nep_get_error). */
    const char* get_error() const {
        if (!fn_get_error_ || !nep_handle_) return "no error";
        return fn_get_error_(nep_handle_);
    }

    char* license_info() {
        return fn_license_info_(nep_handle_);
    }

    /* ── Potential Loading ────────────────────────────────────────────── */

    int read_potential(const char* filename, bool is_rank_0,
                       int rank_id, int device_id, int ff_id)
    {
        int rc = fn_read_potential_(nep_handle_, filename,
                                     is_rank_0 ? 1 : 0,
                                     rank_id, device_id, ff_id);
        if (rc == 0) {
            // Cache frequently-accessed parameters
            rc_radial_ = fn_get_rc_radial_(nep_handle_);
            zbl_enabled_ = fn_get_zbl_enabled_(nep_handle_) != 0;
            zbl_rc_outer_ = fn_get_zbl_rc_outer_(nep_handle_);
            int count = 0;
            const int* elems = fn_get_element_list_(nep_handle_, &count);
            element_list_.assign(elems, elems + count);
        }
        return rc;
    }

    /* ─── Type Mapping ────────────────────────────────────────────────── */

    int set_atom_type_map(int ntypes, const int* type_list) {
        return fn_set_atom_type_map_(nep_handle_, ntypes, type_list);
    }

    /* ─── Core Compute ────────────────────────────────────────────────── */

    int compute(
        int eflag_global, int eflag_atom, int vflag_either,
        int vflag_global, int vflag_atom, int cvflag_atom,
        int nall, int inum, int nlocal,
        int max_neighbors, int num_neighbors,
        int* itype, int* ilist, int* numneigh, int* firstneigh,
        double* position,
        double* potential_per_atom_lmp,
        double* potential_per_atom_copy,
        double* force_per_atom_lmp,
        double* force_per_atom_copy,
        double* virial6_per_atom,
        double* virial9_per_atom,
        double* h_etot_virial_global,
        bool neighbor_rebuilt)
    {
        return fn_compute_(nep_handle_,
                            eflag_global, eflag_atom, vflag_either,
                            vflag_global, vflag_atom, cvflag_atom,
                            nall, inum, nlocal,
                            max_neighbors, num_neighbors,
                            itype, ilist, numneigh, firstneigh,
                            position,
                            potential_per_atom_lmp,
                            potential_per_atom_copy,
                            force_per_atom_lmp,
                            force_per_atom_copy,
                            virial6_per_atom,
                            virial9_per_atom,
                            h_etot_virial_global,
                            neighbor_rebuilt ? 1 : 0);
    }

    /* ─── Memory Management ───────────────────────────────────────────── */

    int reset_data(int inum, int n_local, int n_all, int vflag_either) {
        return fn_reset_data_(nep_handle_, inum, n_local, n_all, vflag_either);
    }

    int free_data() {
        return fn_free_data_(nep_handle_);
    }

    bool get_gpu_memory_stats(size_t& total, size_t& used, size_t& free_mem) {
        return fn_get_gpu_memory_stats_(nep_handle_, &total, &used, &free_mem) != 0;
    }

    int calculate_max_atoms(int mn_radial, int mn_angular,
                            int num_neigh, int ann_dim,
                            int nlocal, int nprocs_total) {
        return fn_calculate_max_atoms_(nep_handle_, mn_radial, mn_angular,
                                        num_neigh, ann_dim, nlocal, nprocs_total);
    }

    /* ─── Parameter Accessors (cached after read_potential) ───────────── */

    double get_rc_radial() const { return rc_radial_; }
    int get_n_max_radial() const { return fn_get_n_max_radial_(nep_handle_); }
    int get_n_max_angular() const { return fn_get_n_max_angular_(nep_handle_); }
    int get_mn_radial() const { return fn_get_mn_radial_(nep_handle_); }
    int get_mn_angular() const { return fn_get_mn_angular_(nep_handle_); }
    int get_ann_dim() const { return fn_get_ann_dim_(nep_handle_); }
    int get_num_types() const { return fn_get_num_types_(nep_handle_); }
    bool get_zbl_enabled() const { return zbl_enabled_; }
    double get_zbl_rc_outer() const { return zbl_rc_outer_; }

    const std::vector<int>& get_element_list() const { return element_list_; }

private:
    void* lib_handle_;
    nep_handle_t* nep_handle_;

    // Function pointers
    nep_create_t                  fn_create_ = nullptr;
    nep_destroy_t                 fn_destroy_ = nullptr;
    nep_get_error_t               fn_get_error_ = nullptr;
    nep_license_load_t            fn_license_load_ = nullptr;
    nep_license_info_t            fn_license_info_ = nullptr;
    nep_read_potential_t          fn_read_potential_ = nullptr;
    nep_set_atom_type_map_t       fn_set_atom_type_map_ = nullptr;
    nep_compute_t                 fn_compute_ = nullptr;
    nep_reset_data_t              fn_reset_data_ = nullptr;
    nep_free_data_t               fn_free_data_ = nullptr;
    nep_get_gpu_memory_stats_t    fn_get_gpu_memory_stats_ = nullptr;
    nep_calculate_max_atoms_t     fn_calculate_max_atoms_ = nullptr;
    nep_get_rc_radial_t           fn_get_rc_radial_ = nullptr;
    nep_get_n_max_radial_t        fn_get_n_max_radial_ = nullptr;
    nep_get_n_max_angular_t       fn_get_n_max_angular_ = nullptr;
    nep_get_mn_radial_t           fn_get_mn_radial_ = nullptr;
    nep_get_mn_angular_t          fn_get_mn_angular_ = nullptr;
    nep_get_ann_dim_t             fn_get_ann_dim_ = nullptr;
    nep_get_num_types_t           fn_get_num_types_ = nullptr;
    nep_get_zbl_enabled_t         fn_get_zbl_enabled_ = nullptr;
    nep_get_zbl_rc_outer_t        fn_get_zbl_rc_outer_ = nullptr;
    nep_get_element_list_t        fn_get_element_list_ = nullptr;

    std::string error_msg_;
    bool loaded_ = false;

    // Cached parameters
    double rc_radial_ = 0.0;
    bool zbl_enabled_ = false;
    double zbl_rc_outer_ = 0.0;
    std::vector<int> element_list_;
};

#endif // NEP_GPU_LOADER_H
