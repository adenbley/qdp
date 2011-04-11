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

    $ret.="    QDPExpr<RHS, OLattice<T1> > rhs(".$rhs.");\n";
    $ret.="    OLattice<T> dest;\n";
    $ret.="    ".$pretty{"partOp"}." op".argsempty($pretty{"partOp"}).";\n";
    $ret.="    Subset s;\n";
    $ret.="    dest.setF(ival->dest);";
    $ret.="    evaluate( dest , op , rhs , s );\n";

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

__global__ void kernel(CUDA_iface_eval * ival)
{
$spucode;
}

extern "C" void function_host(void * ptr)
{
    cout << "function_host()" << endl;

    CUDA_iface_eval * ival = static_cast<CUDA_iface_eval *>(ptr);

    cout << ival->opMetaSize << endl;

    dim3  threads( 320 , 1, 1);
    kernel<<< 1 , threads >>>(ival);
    cudaError_t kernel_call = cudaGetLastError();
    cout << "kernel call: " << string(cudaGetErrorString(kernel_call)) << endl;
}

END
    ;
    close(SPU);
}


