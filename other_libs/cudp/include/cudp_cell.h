#ifndef QDP_CELL_H
#define QDP_CELL_H

#define SPU_CMD_QDPEXPR 1
#define SPU_CMD_SHIFT   2
#define SPU_CMD_PRETTY  3


namespace QDP {

  inline size_t number_block_access( const size_t& sizeT )
  {
    size_t help = 1;
    while ( (sizeT*help) % ( mplcell::spu_min_dma_transfer_size ) )
      help++;
    return help;
    /* switch (sizeT) */
    /*   { */
    /*   case 1: return 128; */
    /*   case 2: return 64; */
    /*   case 4: return 32; */
    /*   case 8: return 16; */
    /*   case 16: return 8; */
    /*   case 24: return 16; */
    /*   case 32: return 4; */
    /*   case 40: return 16;   */
    /*   case 48: return 8; */
    /*   case 144: return 8; */
    /*   default: */
    /* 	size_t help = 1; */
    /* 	while ( (sizeT*help) % 128 ) */
    /* 	  help++; */
    /* 	return help; */
    /*   } */
  }

}


#endif

