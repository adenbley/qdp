AC_INIT([cudp],[0.0.0],[frank.winter@ed.ac.uk])

AC_CONFIG_AUX_DIR(config)
AC_CONFIG_HEADERS([include/cudp_config_internal.h])

AC_LANG([C++])
AC_PROG_CXX(g++ cxx CC)
AC_PROG_RANLIB
AC_CHECK_TOOL(AR, ar, [ar])
AC_CHECK_TOOL(PERL, perl, [perl])

AM_INIT_AUTOMAKE

AC_CONFIG_FILES([
Makefile
include/Makefile
scripts/Makefile
])

AC_OUTPUT

