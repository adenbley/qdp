cuPETE_HDRS =  cuPETE/cuCombiners.h \
	       cuPETE/cuFunctors.h \
              cuPETE/cuPETE.h \
             cuPETE/cuTreeNodes.h \
             cuPETE/cuCreateLeaf.h \
             cuPETE/cuForEach.h \
             cuPETE/cuForEachInOrder.h \
             cuPETE/cuOperatorTags.h \
             cuPETE/cuScalar.h \
             cuPETE/cuTypeComputations.h

# Generic optimization headers
genericdir = cu_scalarsite_generic

GENERIC_HDRS = cudp_scalarsite_generic.h \
	$(genericdir)/generic_addvec.h \
	$(genericdir)/generic_adj_mat_vec.h \
	$(genericdir)/generic_mat_vec.h \
	$(genericdir)/generic_mult_aa.h \
	$(genericdir)/generic_mult_an.h \
	$(genericdir)/generic_mult_na.h \
	$(genericdir)/generic_mult_nn.h \
	$(genericdir)/generic_blas_local_sumsq.h \
	$(genericdir)/generic_blas_vadd.h  \
	$(genericdir)/generic_blas_vaxmby3.h \
	$(genericdir)/generic_blas_vaxmby3_norm.h \
	$(genericdir)/generic_blas_vaxmy3.h \
	$(genericdir)/generic_blas_vaxpby3.h \
	$(genericdir)/generic_blas_vaxpy3.h \
	$(genericdir)/generic_blas_vaxpy3_norm.h \
	$(genericdir)/generic_blas_vscal.h \
	$(genericdir)/generic_blas_vsub.h \
	$(genericdir)/generic_blas_vcscal.h \
	$(genericdir)/generic_blas_vcaxpy3.h \
	$(genericdir)/generic_blas_vcaxmy3.h \
	$(genericdir)/generic_blas_vcaxpby3.h \
	$(genericdir)/generic_blas_vcaxmby3.h \
	$(genericdir)/generic_blas_local_vcdot.h \
	$(genericdir)/generic_blas_local_vcdot_real.h \
	$(genericdir)/qdp_scalarsite_generic_linalg.h \
	$(genericdir)/qdp_scalarsite_generic_blas.h \
	$(genericdir)/qdp_scalarsite_generic_cblas.h \
	$(genericdir)/qdp_scalarsite_generic_blas_g5.h \
	$(genericdir)/generic_blas_vadd3_g5.h \
	$(genericdir)/generic_blas_vscal_g5.h \
	$(genericdir)/generic_blas_vaxpy3_g5.h \
	$(genericdir)/generic_blas_vaypx3_g5.h \
	$(genericdir)/generic_blas_vaxpby3_g5.h \
	$(genericdir)/generic_blas_g5.h \
	$(genericdir)/generic_mv_switchbox.h \
	$(genericdir)/generic_spin_proj_inlines.h \
	$(genericdir)/generic_spin_recon_inlines.h \
	$(genericdir)/generic_spin_proj.h \
	$(genericdir)/generic_spin_recon.h \
	$(genericdir)/generic_fused_spin_proj.h \
	$(genericdir)/generic_fused_spin_recon.h \
	$(genericdir)/qdp_generic_spin_project_evaluates.h \
	$(genericdir)/qdp_generic_fused_spin_proj_evaluates.h \
	$(genericdir)/qdp_generic_fused_spin_recon_evaluates.h \
	$(genericdir)/generic_spin_aggregate.h \
	$(genericdir)/qdp_scalarsite_generic_blas_wrapper.h \
	$(genericdir)/qdp_scalarsite_generic_cblas_wrapper.h \
	$(genericdir)/qdp_scalarsite_generic_blas_g5_wrapper.h \
	$(genericdir)/qdp_generic_spin_project_evaluates_wrapper.h \
	$(genericdir)/qdp_generic_fused_spin_proj_evaluates_wrapper.h \
	$(genericdir)/qdp_generic_fused_spin_recon_evaluates_wrapper.h \
	$(genericdir)/qdp_scalarsite_generic_linalg_wrapper.h


# QCDOC NONSPECIFIC BAGEL stuff
bageldir = cu_scalarsite_bagel_qdp
BAGEL_HDRS = cudp_scalarsite_bagel_qdp.h \
	$(bageldir)/qdp_scalarsite_bagel_qdp_linalg.h \
	$(bageldir)/qdp_scalarsite_bagel_qdp_blas.h \
	$(bageldir)/qdp_scalarsite_bagel_qdp_blas_g5.h

# HEADERS for the memory allocator
MEMORY_HDRS = cudp_allocator.h \
	      cudp_singleton.h \
	      cudp_default_allocator.h \
	      cudp_qcdoc_allocator.h

# All the include files - avoid flattening of dirs by using nobase
nobase_include_HEADERS = \
		cudp_config.h \
		cudp_forward.h \
		cudp_globalfuncs.h \
		cudp_inner.h \
		cudp_init.h \
		cudp_io.h \
		cudp_qlimewriter.h \
		cudp_stdio.h \
		cudp_layout.h \
		cudp_map.h \
		cudp_multi.h \
		cudp_newops.h \
		cudp_optops.h \
		cudp_outer.h \
		cudp_outersubtype.h \
		cudp_params.h \
		cudp_precision.h \
		cudp_primcolormat.h \
		cudp_primcolorvec.h \
		cudp_primgamma.h \
		cudp_primitive.h \
		cudp_primmatrix.h \
		cudp_primscalar.h \
		cudp_primseed.h \
		cudp_primspinmat.h \
		cudp_primspinvec.h \
		cudp_primvector.h \
		cudp_profile.h \
        cudp_stopwatch.h \
		cudp_flopcount.h \
		cudp_iogauge.h \
		cudp_crc32.h \
		cudp_byteorder.h \
		cudp_util.h \
		cudp_xmlio.h \
		cudp_qdpexpr.h \
		cudp.h \
		cudp_qdpio.h \
		cuQDPOperators.h \
		cudp_qdpsubtype.h \
		cudp_qdptype.h \
		cudp_random.h \
		cudp_rannyu.h \
		cudp_reality.h \
		cudp_simpleword.h \
		cudp_specializations.h \
		cudp_subset.h \
		cudp_traits.h \
		cudp_word.h \
        cudp_iface.h \
		cudp_dispatch.h \
	    cudp_scalar_specific.h \
	    cudp_parscalar_specific.h \
	    cudp_scalarvec_specific.h \
	    cudp_parscalarvec_specific.h \
	 	cudp_defs.h \
		cudp_scalarsite_defs.h \
		cudp_scalarvecsite_defs.h \
		cudp_strnlen.h \
		cudp_db.h \
		cudp_db_imp.h \
		cudp_db_stub.h \
		cudp_map_obj.h \
		cudp_map_obj_null.h \
		cudp_map_obj_memory.h \
		cudp_map_obj_disk.h \
		cudp_disk_map_slice.h \
        $(cuPETE_HDRS) \
        $(GENERIC_HDRS) \
		$(MEMORY_HDRS) \
		$(BAGEL_HDRS)

#nodist_include_HEADERS = qdp_config_internal.h


