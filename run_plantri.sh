if [ ! -f /plantri_compiled ]; then
    gcc plantri55/plantri.c -o plantri_compiled
fi

for i in {6..25}
do
   ./plantri_compiled -b $i -d all_graphs/graphs_out_$i
done

# ./plantri_compiled -b 11 -d all_graphs/graphs_out_11