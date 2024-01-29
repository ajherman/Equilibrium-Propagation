#!/usr/bin/perl

$queue = `squeue -u rstrauss`;

my $nlines = $queue =~ tr/\n//;

if ($nlines < 3 and $nlines > 1){
    if ($queue =~ /(cn\d+)/){
        my $dev = $1;
    
        print "ssh $dev \"$ARGV[0]\"";
        my $out = `ssh $dev "$ARGV[0]"`;
        print $out;
    }
}
