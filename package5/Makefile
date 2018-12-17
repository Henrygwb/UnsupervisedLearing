all: labelsw labelsw_bunch2 

labelsw: align.h labelsw.c matrix.o simplex.o align.o
#
	gcc -o labelsw -g labelsw.c matrix.o simplex.o align.o -lm

labelsw_bunch2: align.h labelsw_bunch2.c matrix.o simplex.o align.o
#
	gcc -o labelsw_bunch2 -g labelsw_bunch2.c matrix.o simplex.o align.o -lm

matrix.o: matrix.c
	gcc -g -c matrix.c

align.o: align.c
	gcc -g -c align.c

simplex.o: simplex.c
	gcc -g -c simplex.c
