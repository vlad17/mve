d="$1"
w="$2"
for e in 1 10 100 ; do
en=""
for lr in 2 3 4 ; do
for pn in "" --param_noise ; do
en="$en data/ddpg_e${e}_lr${lr}_d${d}_w${w}${pn}_hc-hard:AverageReturn:${lr}lr${pn}"
done
done
echo $en
python mpc_bootstrap/plot.py $en --outfile "ar-e${e}-d${d}-w${w}.pdf" --yaxis "rollout returns" --notex
done
