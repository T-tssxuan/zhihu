awk '{ if (NR > 10) print $0}' ../nohup.out > tmp
awk '{ if (NR % 3 == 2) print $0}' tmp > tmp1
awk 'BEGIN{FS=","} {print $2}' tmp1 > tmp2
awk 'BEGIN{FS=": "} {print $2}' tmp2 > data
rm tmp*
