## Kokkos Staging Spaces

Kokkos Staging Spaces (KSS) adds inter-application data exchange support to Kokkos parallel programming model. It uses Dataspaces2 (https://github.com/Zhang690683220/dspaces/tree/kokkos) as the backend. KSS leverages in-memory stage in&out and unburdens the programmer from designing ad hoc data redistribution between applications in the complex scientific workflow.

*Note: Kokkos Staging Spaces is in an experimental development stage.*

## Dependencies

* MPI
* Kokkos (https://github.com/kokkos/kokkos)
* Dataspaces2 (https://github.com/Zhang690683220/dspaces/tree/kokkos)

## Build
````bash
> cmake ${KSS_SOURCE_DIR} \
    -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
    -DDATASPACES_ROOT=${DATASPACES_INSTALL_PREFIX} \
    -DCMAKE_CXX_COMPILER=${CXX}
````

## API

````C++
/**
 * @brief Initialize the Kokkos::StagingSpace
 * 
 * Required when using Kokkos::StagingSpace.
 * Call after Kokkos::initialize()
 * 
 */
Kokkos::Staging::initialize();

/**
 * @brief Views whose memory space is in remote staging server
 * 
 * Same usage as other Kokkos::View. Support up to 8 dimension. 
 * 
 */
Kokkos::View<Data_t*, Kokkos::StagingSpace> v_S1("ViewStaging1D", arg_N1);
Kokkos::View<Data_t**, Kokkos::LayoutLeft, Kokkos::StagingSpace> v_S2("ViewStaging2D", arg_N1, arg_N2);

/**
 * @brief Set version of the staging view
 * 
 * Default version is set to be 0
 * 
 * @param[in] dst: staging view to be set
 * @param[in] version specified version 
 *
 */
Kokkos::Staging::set_version(const View<DT, DP...>& dst, size_t version);

/**
 * @brief Bind two staging views in different layout
 * 
 * Required as the pre-requisite for DeepCopy between views in different layout
 * 
 * @param[in] dst: staging view in source layout
 * @param[in] client staging view in dstnation layout
 *
 */
Kokkos::Staging::view_bind_layout(const View<DT, DP...>& dst, const View<ST, SP...>& src);

/**
 * @brief Finalize the Kokkos::StagingSpace
 * 
 * Required before Kokkos::finalize()
 * 
 */
Kokkos::Staging::finalize();
````

## Example 1: DeepCopy between views in same layout
````C++
using ViewHost_t    = Kokkos::View<Data_t**, Kokkos::HostSpace>;
using ViewStaging_t = Kokkos::View<Data_t**, Kokkos::StagingSpace>;

ViewHost_t v_P("PutView", i1, i2);
ViewStaging_t v_S("StagingView", i1, i2);
ViewHost_t v_G("GetView", i1, i2);

// from host to staging
Kokkos::deep_copy(v_S, v_P);

// from staging to host   
Kokkos::deep_copy(v_G, v_S);

````

## Example 2: DeepCopy between views in different layout
````C++
using ViewHost_lr_t    = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using ViewHost_ll_t    = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using ViewStaging_lr_t = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::StagingSpace>;
using ViewStaging_ll_t = Kokkos::View<double**, Kokkos::LayoutLeft,Kokkos::StagingSpace>;


ViewHost_lr_t v_P("PutView", i1, i2);
ViewStaging_lr_t v_S_lr("StagingView_LayoutRight", i1, i2);

ViewStaging_ll_t v_S_ll("StagingView_LayoutLeft", i1, i2);
ViewHost_ll_t v_G("GetView", i1, i2);

// bind two staging views in different layout
Kokkos::Staging::view_bind_layout(v_S_ll, v_S_lr);

// from host to staging
Kokkos::deep_copy(v_S_lr, v_P);

// from staging to host   
Kokkos::deep_copy(v_G, v_S_ll);

````


