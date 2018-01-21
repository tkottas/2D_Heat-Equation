#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define NXPROB      1008               /* x dimension of problem grid */
#define NYPROB      1008               /* y dimension of problem grid */
#define STEPS      100              /* number of time steps */
#define MAXWORKER   16                  /* maximum number of worker tasks */
#define MINWORKER   1                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define UTAG        4                  /* message tag */
#define DTAG        5                  /* message tag */ 
#define NONE        -1                  /* indicates no neighbor */
#define DONE        10                 /* message tag */
#define MASTER      0                  /* taskid of first process */
#define OPENMPTHR   4

#include <omp.h>


struct Parms { 
    float cx;
    float cy;
} parms = {0.1, 0.1};

int main (int argc, char *argv[]) {

    void inidat(), prtdat(), updateinner(),updateouter();
    int taskid,square_root,                     /* this task's unique id */
    numtasks,                   /* number of tasks */
    averblock,blocks_s,offset,extra,   /* for sending blocks_s of data */
    dest, source,               /* to - from for message send-receive */
    left,right,up,down,        /* neighbor tasks */
    msgtype,                    /* for message types */
    rc,start,end,               /* misc */
    i,j,checki,checkj,ix,iy,iz,it,sum1,sum2,flag=0,allflag=1,mpisupport;              /* loop variables */
    MPI_Status status;
    MPI_Datatype sendtype_col,sendtype_row,sendarray,receivearray;
 	MPI_Comm comm;
	MPI_Request sendrequests[4],receiverequests[4],finalsend;
    double starttime, endtime;
    float u[NXPROB][NYPROB];      
        int curr_task;
        int offsetx=0;
        int offsety=0;
        MPI_Request stats;
 	
	MPI_Init(&argc,&argv);
    omp_set_dynamic(0);
    omp_set_num_threads(OPENMPTHR);
    starttime = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    for(square_root=1;square_root<=7;square_root++)
        if(square_root*square_root==numtasks) break;

    MPI_Request finalsends[numtasks];
    int dim[2]={square_root,square_root};
    int period[2] ={0,0};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, 1, &comm);

    int num_of_blocks = NYPROB/square_root;
    blocks_s = NYPROB/square_root;
    
    MPI_Type_vector(blocks_s+2, 1, blocks_s+2, MPI_FLOAT, &sendtype_col);
    MPI_Type_commit(&sendtype_col);
 	
	MPI_Type_vector(blocks_s+2, 1, 1, MPI_FLOAT, &sendtype_row);
    MPI_Type_commit(&sendtype_row);
 	
    
    int big_size[2] ={blocks_s+2,blocks_s+2};
   	int small_size[2] ={blocks_s,blocks_s};
   	int grid_size[2]={NXPROB,NYPROB};
    int new_arrstart[2] ={1,1};
    //int grid_start[2]={0,0}; idio me to period
 	
 
	MPI_Type_create_subarray(2, big_size,small_size,new_arrstart,MPI_ORDER_C, MPI_FLOAT, &sendarray);
    MPI_Type_commit(&sendarray); 

    MPI_Type_create_subarray(2, grid_size, small_size, period,MPI_ORDER_C, MPI_INT, &receivearray);
    MPI_Type_commit(&receivearray); 

    float  localu[2][blocks_s+2][blocks_s+2];    

    for (i = 0; i <=blocks_s+1; i++) 
        for (j = 0; j <= blocks_s+1; j++){
            localu[0][i][j] = (float)(i*(blocks_s-i-1)*j*(blocks_s-j-1));
            localu[1][i][j] = (float)(i*(blocks_s-i-1)*j*(blocks_s-j-1));
        }
   
     
          
    iz = 0;

    MPI_Cart_shift(comm,1,1,&left,&right);
    MPI_Cart_shift(comm,0,1,&up,&down);
    
    MPI_Send_init(&localu[iz][1][0] , 1, sendtype_row, up, DTAG, MPI_COMM_WORLD, &sendrequests[0] );
    MPI_Send_init(&localu[iz][blocks_s][0],1, sendtype_row, down, UTAG, MPI_COMM_WORLD,&sendrequests[1]);
 	MPI_Send_init(&localu[iz][0][1], 1 , sendtype_col,left, LTAG, MPI_COMM_WORLD,&sendrequests[2]);
 	MPI_Send_init(&localu[iz][0][blocks_s], 1 , sendtype_col, right, RTAG, MPI_COMM_WORLD,&sendrequests[3]);

    MPI_Recv_init(&localu[iz][0][0], 1, sendtype_row, up, UTAG , MPI_COMM_WORLD, &receiverequests[0]);
	MPI_Recv_init(&localu[iz][blocks_s+1][0],1, sendtype_row, down, DTAG, MPI_COMM_WORLD, &receiverequests[1]);
	MPI_Recv_init(&localu[iz][0][0], 1, sendtype_col, left,RTAG , MPI_COMM_WORLD, &receiverequests[2]);
    MPI_Recv_init(&localu[iz][0][blocks_s+1], 1, sendtype_col, right, LTAG, MPI_COMM_WORLD, &receiverequests[3]);
  

   for (it = 1; it <= STEPS; it++){

        MPI_Startall(4,sendrequests);
        MPI_Startall(4,receiverequests);
        updateinner(blocks_s,blocks_s+2,&localu[iz][0][0],&localu[1-iz][0][0]);	
    	MPI_Waitall(4,receiverequests,MPI_STATUSES_IGNORE);
        updateouter(up,down,left,right,blocks_s,blocks_s+2,taskid,num_of_blocks,&localu[iz][0][0],&localu[1-iz][0][0]);			
        MPI_Waitall(4,sendrequests,MPI_STATUSES_IGNORE);
        iz=1 - iz;

    }

    if(numtasks==1){

   	 	MPI_Type_free(&sendarray);
	    MPI_Type_free(&sendtype_col);
	    MPI_Type_free(&sendtype_row);
	    MPI_Type_free(&receivearray);
 		endtime = MPI_Wtime();
		printf("That took %f seconds\n",endtime-starttime);
        MPI_Finalize();
           return;
 	}

	MPI_Isend(&(localu[iz][0][0]), 1, sendarray, 0, DONE, MPI_COMM_WORLD,&finalsend);

     if (taskid == MASTER) {
     	
        for(curr_task=0;curr_task<numtasks;curr_task++){
        	MPI_Irecv(&(u[offsetx][offsety]), 1, receivearray, curr_task, DONE, MPI_COMM_WORLD, &finalsends[curr_task]);
        	if(offsety+blocks_s>NYPROB-1){
        		offsety=0;
        		offsetx+=blocks_s;
        	}else
        		offsety+=blocks_s;
        	
        }
        MPI_Waitall(numtasks, finalsends,MPI_STATUSES_IGNORE);

	}

	MPI_Wait(&finalsend,MPI_STATUS_IGNORE);
	
    MPI_Type_free(&sendarray);
    MPI_Type_free(&sendtype_col);
    MPI_Type_free(&sendtype_row);
    MPI_Type_free(&receivearray);
    endtime = MPI_Wtime();
    printf("That took %f seconds\n",endtime-starttime);
    MPI_Finalize();
 
}



/**************************************************************************
 *  subroutine updateinner
 ****************************************************************************/
void updateinner(int sizep,int size, float *u1, float *u2) {

    int ix, iy;
    #pragma omp parallel
    {
        int chunk=sizep/omp_get_num_threads();
        #pragma omp for schedule(static,chunk) private(iy) nowait
        for (ix = 2; ix < sizep; ix++){
            for (iy = 2; iy < sizep; iy++){
                *(u2+ix*size+iy) = *(u1+ix*size+iy)  + 
                                parms.cx * (*(u1+(ix+1)*size+iy) +
                                *(u1+(ix-1)*size+iy) - 
                                2.0 * *(u1+ix*size+iy)) +
                                parms.cy * (*(u1+ix*size+iy+1) +
                                *(u1+ix*size+iy-1) - 
                                2.0 * *(u1+ix*size+iy));
            }
        }
    }
}


/**************************************************************************
 *  subroutine updateouter
 ****************************************************************************/
void updateouter(int up,int down,int left,int right,int sizep, int size, int task,int nblocks, float *u1, float *u2) {

    int ix,iy,lines_per_thread;
   	#pragma omp parallel
        {
        if (up != NONE){
                int chunk=sizep/omp_get_num_threads(); 
                #pragma omp for schedule(static,chunk) private(iy) nowait
        	for(ix=size+1;ix<=2*size-2;ix++){
        		*(u2+ix) = *(u1+ix)  + 
            					parms.cx * (*(u1+ix+size) +
                                *(u1+ix-size) - 
                                2.0 * *(u1+ix)) +
                                parms.cy * (*(u1+ix+1) +
                                *(u1+ix-1) - 
                                2.0 * *(u1+ix));
            }
        }
     
        if (down != NONE){
                int chunk=sizep/omp_get_num_threads();
                #pragma omp for schedule(static,chunk) private(iy) nowait
    		for(ix=size*(size-2)+1;ix<=size*size-2;ix++){
        		*(u2+ix) = *(u1+ix)  + 
            					parms.cx * (*(u1+ix+size) +
                                *(u1+ix-size) - 
                                2.0 * *(u1+ix)) +
                                parms.cy * (*(u1+ix+1) +
                                *(u1+ix-1) - 
                                2.0 * *(u1+ix));
            }
        }

        if (left != NONE){
                int chunk=sizep/omp_get_num_threads();
                #pragma omp for schedule(static,chunk) private(iy) nowait
    		for(ix=2*size+1;ix<=size*(size-3)+1;ix+=size){
    			*(u2+ix) = *(u1+ix)  + 
            					parms.cx * (*(u1+ix+size) +
                                *(u1+ix-size) - 
                                2.0 * *(u1+ix)) +
                                parms.cy * (*(u1+ix+1) +
                                *(u1+ix-1) - 
                                2.0 * *(u1+ix));
    		}
        }

        if (right != NONE){
                int chunk=sizep/omp_get_num_threads();
                #pragma omp for schedule(static,chunk) private(iy) nowait
    		for(ix=3*size-2;ix<=size*(size-2)-2;ix+=size){
        		*(u2+ix) = *(u1+ix)  + 
            					parms.cx * (*(u1+ix+size) +
                                *(u1+ix-size) - 
                                2.0 * *(u1+ix)) +
                                parms.cy * (*(u1+ix+1) +
                                *(u1+ix-1) - 
                                2.0 * *(u1+ix));
            }
        }
       }
}
/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int size, float *u) {

int ix, iy;

for (ix = 0; ix <= nx-1; ix++) 
  for (iy = 0; iy <= size-1; iy++)
     *(u+ix*size+iy) = (float)(ix * (nx - ix - 1) * iy * (size - iy - 1));

}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int size, float *u1, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
	for (iy = size-1; iy >= 0; iy--) {
	 	for (ix = 0; ix <= nx-1; ix++) {
	    	fprintf(fp, "%6.1f", *(u1+ix*size+iy));
	    if (ix != nx-1) 
	      	fprintf(fp, " ");
	    else
	      	fprintf(fp, "\n");
	    }
	}
	fclose(fp);
}  