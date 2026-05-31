function wait_until_done {
# loop until some jobs with the given name exist in the queue
    USER=david
    n=1
    nminutes=0
    while [ $n -ne 0 ]
    do
	sleep 60
	nminutes=`expr $nminutes + 1`
	n=`qstat | grep ${USER} | grep $1 | grep -v " C " | wc -l`
	#n=`ps -u${USER} | grep $1 | grep -v grep |  wc -l`
	echo "${1}: waiting ${nminutes} minutes, still ${n} jobs running ..."
    done
}                                                                                                                           

function wait_until_server_ok {
  qstat -B
  EXITCODE=$?
  while [ $EXITCODE -ne 0 ]
  do
    echo "PBS Server not OK, waiting 1 minute"
    sleep 60
    qstat -B 
    EXITCODE=$? 
  done
}
