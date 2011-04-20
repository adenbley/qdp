#!/usr/bin/env perl

use Switch;


    $cparens = qr/
    (?<intern>
      \<
      (?:
        [^<>]++
        |
        (?&intern)
      )*+
      \>
    )
    /x;



sub parse_template
{
    my ($expr) = @_;
    local($cparens);
    local($tmp);
    local(@ret);

    if ($expr !~ /\</) {
	push(@ret,$expr);
	return @ret;
    }

    $cparens = qr/
    (?<intern>
      \<
      (?:
        [^<>]++
        |
        (?&intern)
      )*+
      \>
    )
    /x;

    if ($expr =~ m/
    (?'root'[^<>]++)
    \<
    (?'part1'[^<>,]++(?:$cparens)?\s*)
    (?'part2'\,[^<>,]++(?:$cparens)?\s*)?
    (?'part3'\,[^<>,]++(?:$cparens)?\s*)?
    (?'part4'\,[^<>,]++(?:$cparens)?\s*)?
    \>
    /x)
    {
	if (defined $+{root}) {
	    print "root=".$+{root}."\n";
	    $tmp=$+{root};
	    $tmp=~s/^\s*|\s*$//g;
	    push(@ret,$tmp);
	}
	
	if (defined $+{part1}) {
	    print "part1=".$+{part1}."\n";
	    $tmp=$+{part1};
	    $tmp=~s/^\s*|\s*$//g;
	    push(@ret,$tmp);
	}

	if (defined $+{part2}) {
	    print "part2=".$+{part2}."\n";
	    $tmp=$+{part2};
	    $tmp=~s/^\,(.*+)$/$1/;
	    $tmp=~s/^\s*|\s*$//g;
	    push(@ret,$tmp);
	}

	if (defined $+{part3}) {
	    print "part3=".$+{part3}."\n";
	    $tmp=$+{part3};
	    $tmp=~s/^\,(.*+)$/$1/;
	    $tmp=~s/^\s*|\s*$//g;
	    push(@ret,$tmp);
	}

	if (defined $+{part4}) {
	    print "part4=".$+{part4}."\n";
	    $tmp=$+{part4};
	    $tmp=~s/^\,(.*+)$/$1/;
	    $tmp=~s/^\s*|\s*$//g;
	    push(@ret,$tmp);
	}
    } else {
	print "could not match template $_[0]\n";
    }
    return @ret;
}


$varcnt=0;
$tmpcnt=0;


sub args
{
    my ($fn) = @_;
    local($ret);
    if ($fn =~ /QDP::.*Vector/) {
	$ret="(0)";
    } elsif ($fn =~ /QDP::.*Matrix/) {
	$ret="(0,0)";
    } else {
	$ret="()";
    }
    return $ret;
}

sub argsempty
{
    my ($fn) = @_;
    local($ret);
    if ($fn =~ /QDP::.*Vector/) {
	$ret="(0)";
    } elsif ($fn =~ /QDP::.*Matrix/) {
	$ret="(0,0)";
    } else {
	$ret="";
    }
    return $ret;
}


sub recu2
{
    my ($expr) = @_;
    local(@parts,$ret);

    print "\nrecu2 entered with $expr\n";

    @parts = parse_template($expr);
    $"="\n";
    print "parts found:\n";
    print "@parts\n";

    if ($parts[0] =~ /QDP::BinaryNode/) { 
	print "BINARY found:\n";
	$ret=$expr."(";
	$ret.=$parts[1].args($parts[1]);
	if ($#parts > 1) {
	    $ret.=",";
	    $ret.=recu2($parts[2]);
	}
	if ($#parts > 2) {
	    $ret.=",";
	    $ret.=recu2($parts[3]);
	}
	$ret.=")";
    } 
    elsif ($parts[0] =~ /QDP::TrinaryNode/) { 
	print "TRINARY found:\n";
	$ret=$expr."(";
	#$ret.=$parts[1]."()";
	$ret.=$parts[1].args($parts[1]);
	if ($#parts > 1) {
	    $ret.=",";
	    $ret.=recu2($parts[2]);
	}
	if ($#parts > 2) {
	    $ret.=",";
	    $ret.=recu2($parts[3]);
	}
	if ($#parts > 3) {
	    $ret.=",";
	    $ret.=recu2($parts[4]);
	}
	$ret.=")";
    }
    elsif ($parts[0] =~ /QDP::UnaryNode/) { 
	print "UNARY found:\n";
	$ret=$expr."(";
	#$ret.=$parts[1]."()";
	$ret.=$parts[1].args($parts[1]);
	if ($#parts > 1) {
	    $ret.=",";
	    $ret.=recu2($parts[2]);
	}
	$ret.=")";
    }
    elsif ($parts[0] =~ /QDP::GammaConst/) { 
	print "GammaConst found!\n";
	$ret.="QDP::GammaConst<".$parts[1].", ".$parts[2].">()";
    }
    elsif ($parts[0] =~ /QDP::GammaType/) { 
	print "GammaType found!\n";
	$ret.="QDP::GammaType<".$parts[1].">(0)";
    }
    elsif ($parts[0] =~ /QDP::OScalar/) { 
	if ($parts[1] =~ /QDP::PScalar<QDP::PScalar<QDP::RScalar/) { 
	    print "QDP::OScalar<QDP::PScalar<QDP::PScalar<QDP::RScalar... found!\n";
	    $ret.="0";
	}
    }
    elsif ($parts[0] =~ /QDP::Reference/) {
	print "REFERENCE found:\n";
	local(@partsrec);
	@partsrec = parse_template($parts[1]);
	if ( $partsrec[0]=~/QDP\:\:QDPType/ ) {
	    if ( $partsrec[2]=~/QDP\:\:OLattice/ ) {
		push(@defines  ,"    $partsrec[2] \t a$varcnt;");
		$type=$partsrec[2];
		$type=~ s/.*<.*<.*<.*<(.*)>.*>.*>.*>.*/$1/g;
		push(@definesno,"    $partsrec[2] \t a$varcnt;\n    fill<$type>(a$varcnt);");
		$ret.="a$varcnt";
		$varcnt++;
		$tmpcnt++;
	    }
	    elsif ( $partsrec[2]=~/QDP\:\:OScalar/ ) {
		push(@defines  ,"    $partsrec[2] \t a$varcnt;");
		$type=$partsrec[2];
		$type=~ s/.*<.*<.*<.*<(.*)>.*>.*>.*>.*/$1/g;
		push(@definesno,"    $partsrec[2] \t a$varcnt;\n    fill<$type>(a$varcnt);");
		$ret.="a$varcnt";
		$varcnt++;
	    }
	} else {
	    print "internal error: QDP::Reference not followed by QDP::QDPType giving up!\n";
	}
    }
    else { 
	print "don't know this root element\n";
    }
    

    return $ret;
}


sub parse_pretty
{
    my ($pretty) = @_;

    local(%ret);

    if ($pretty =~ m/
      ^[^\[\]]++
      \[with\s
      T\s\=\s(?'partT'[^<>,]++(?:$cparens)?)\,\s
      T1\s\=\s(?'partT1'[^<>,]++(?:$cparens)?)\,\s
      Op\s\=\s(?'partOp'[^<>,]++(?:$cparens)?)\,\s
      RHS\s\=\s(?'partRHS'[^<>,]++(?:$cparens)?)
      \]$
    /x)
    {
	print "pretty matched\n";

	if (defined $+{partT}) {
	    $ret{"partT"}=$+{partT};
	}
	if (defined $+{partT1}) {
	    $ret{"partT1"}=$+{partT1};
	}
	if (defined $+{partOp}) {
	    $ret{"partOp"}=$+{partOp};
	}
	if (defined $+{partRHS}) {
	    $ret{"partRHS"}=$+{partRHS};
	}

    } else {
	print "pretty does not match\n";
    }
    print %ret;
    return %ret;
}




sub spu
{
    my ($prettyfunc) = @_;

    local(%pretty,$rhs,$ret);

    %pretty = parse_pretty($prettyfunc);
    $#defines   = -1;
    $#definesno = -1;
    $#maxelem   = -1;
    $varcnt=0;
    $tmpcnt=0;
    $rhs = recu2($pretty{"partRHS"});

    $ret.="    typedef ".$pretty{"partT"}." T;\n";
    $ret.="    typedef ".$pretty{"partT1"}." T1;\n";
    $ret.="    typedef ".$pretty{"partOp"}." Op;\n";
    $ret.="    typedef ".$pretty{"partRHS"}." RHS;\n";

    foreach $def (@maxelem)
    {
	$ret.=$def."\n";
    }
    foreach $def (@defines)
    {
	$ret.=$def."\n";
    }


    $ret.="    #ifdef __CUDA_ARCH__\n";

    $ret.="    QDPExpr<RHS, OLattice<T1> > rhs(".$rhs.");\n";
    $ret.="    OLattice<T> dest;\n";
    $ret.="    ".$pretty{"partOp"}." op".argsempty($pretty{"partOp"}).";\n";
    $ret.="    Subset s;\n";
    $ret.="    s.make( ival->hasOrderedRep , ival->start , ival->end , ival->siteTable );\n";
    $ret.="    dest.setF(ival->dest);\n";
    $ret.="    FlattenTag flattenTag;\n";
    $ret.="    flattenTag.numberLeafs = ival->numberLeafs;\n";
    $ret.="    flattenTag.numberNodes = ival->numberNodes;\n";
    $ret.="    flattenTag.leafDataArray = ival->leafDataArray;\n";
    $ret.="    flattenTag.nodeDataArray = ival->nodeDataArray;\n";
    $ret.="    forEach(rhs, flattenTag , NullCombine());\n";
    $ret.="    evaluate( dest , op , rhs , s );\n";

    $ret.="    #endif\n";

    return($ret);
}



$spufile=$ARGV[0];

@prettys = <STDIN>;

if ($#prettys) {
    print "more than one line found!\n";
    exit(1);
}

foreach $pretty (@prettys)
{
    $pretty=~s/\n//g;
    print ">>>>>>> processing ".$pretty."\n";

    $spucode=spu($pretty);

    open(SPU, ">$spufile");

    print SPU <<END
#include "cudp.h"
#include "cudp_iface.h"
#include <iostream>
using namespace QDP;
using namespace std;

__global__ void kernel(IfaceCudp * ival)
{
$spucode;
}

extern "C" void function_host(void * ptr)
{
    cout << "function_host()" << endl;

    // make a copy of the incoming struct
    IfaceCudp * ival;
    cudaError_t ret;
    ret = cudaMallocHost((void **)(&ival),sizeof(IfaceCudp));
    cout << "cudaMallocHost     " << sizeof(IfaceCudp) << " : " << string(cudaGetErrorString(ret)) << endl;

    ret = cudaMemcpy(ival,ptr,sizeof(IfaceCudp),cudaMemcpyHostToHost);
    cout << "cudaMemcpy to host to host: " << string(cudaGetErrorString(ret)) << endl;

    //IfaceCudp * ival = static_cast<IfaceCudp *>(ptr);
    cout << "dest:" << ival->dest << endl;
    for (int i=0;i<ival->numberLeafs;i++)
	cout << "leaf" << i << " " << ival->leafDataArray[i].pointer << " " << ival->leafDataArray[i].misc << endl;
    for (int i=0;i<ival->numberNodes;i++)
	cout << "node" << i << " " << ival->nodeDataArray[i].pointer << endl;

    FlattenTag::LeafData * save_leafarray = ival->leafDataArray;
    FlattenTag::NodeData * save_nodearray = ival->nodeDataArray;

    IfaceCudp * ival_dev;
    ret = cudaMalloc((void **)(&ival_dev),sizeof(IfaceCudp));
    cout << "get device memory for kernel interface    " << sizeof(IfaceCudp) << " : " << string(cudaGetErrorString(ret)) << endl;

    if (ival->numberLeafs > 0) {
	ret = cudaMalloc((void **)(&ival->leafDataArray),sizeof(FlattenTag::LeafData) * ival->numberLeafs);
	cout << "get device memory for leaf data pointers  " 
	    << sizeof(FlattenTag::LeafData) * ival->numberLeafs 
	    << " : " << string(cudaGetErrorString(ret)) << endl;
	ret = cudaMemcpy(ival->leafDataArray,save_leafarray ,
			 sizeof(FlattenTag::LeafData) * ival->numberLeafs,
			 cudaMemcpyHostToDevice);
	cout << "copy leaf pointers to device:     " 
	    << string(cudaGetErrorString(ret)) << endl;
    }

    if (ival->numberNodes > 0) {
	ret = cudaMalloc((void **)(&ival->nodeDataArray),sizeof(FlattenTag::NodeData) * ival->numberNodes);
	cout << "get device memory for node data pointers  " 
	    << sizeof(FlattenTag::NodeData) * ival->numberNodes 
	    << " : " << string(cudaGetErrorString(ret)) << endl;
	ret = cudaMemcpy(ival->nodeDataArray,save_nodearray ,
			 sizeof(FlattenTag::NodeData) * ival->numberNodes,
			 cudaMemcpyHostToDevice);
	cout << "copy node pointers to device:     " 
	    << string(cudaGetErrorString(ret)) << endl;
    }

    ret = cudaMemcpy(ival_dev,ival,sizeof(IfaceCudp),cudaMemcpyHostToDevice);
    cout << "copy interface to device:         " << string(cudaGetErrorString(ret)) << endl;


    int thr=1024;
    cout << "trying " << thr << " threads/block on " << ival->numSiteTable << " sites" << endl;
    while ( (ival->numSiteTable % thr) && (thr > 32)  ) {
	thr = thr >> 1;
	cout << "trying " << thr << " threads/block on " << ival->numSiteTable << " sites" << endl;
    }
    if (thr >= 32)
	cout << "using " << thr << " threads/block on " << ival->numSiteTable << " sites" << endl;
    else
	cout << "ERROR!" << endl;

    
    bool run_ok=false;
    while (!run_ok) {
	int num_blocks=ival->numSiteTable/thr;
	cout << "launching " << num_blocks << " blocks with " << thr << " threads each..." << endl;
	dim3  blocksPerGrid( num_blocks , 1, 1);
	dim3  threadsPerBlock( thr , 1, 1);

	kernel<<< blocksPerGrid , threadsPerBlock >>>( ival_dev );
	cudaError_t kernel_call = cudaGetLastError();
	cout << "kernel call:                      " << string(cudaGetErrorString(kernel_call)) << endl;

	run_ok = kernel_call == cudaSuccess;
	thr = thr >> 1;
    }

    ret = cudaFreeHost(ival);
    cout << "free memory for host interface:   " << string(cudaGetErrorString(ret)) << endl;

    if (ival->numberLeafs > 0) {
	ret = cudaFree(ival->leafDataArray);
	cout << "free memory for leaf pointers:    " 
	    << string(cudaGetErrorString(ret)) << endl;
    }

    if (ival->numberNodes > 0) {
	ret = cudaFree(ival->nodeDataArray);
	cout << "free memory for node pointers:    " 
	    << string(cudaGetErrorString(ret)) << endl;
    }

    ret = cudaFree(ival_dev);
    cout << "free memory for device interface: " << string(cudaGetErrorString(ret)) << endl;

}

END
    ;
    close(SPU);
}


