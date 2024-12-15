#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits.h>
#include<float.h>
#include <omp.h>

struct values{
  float distance;
  int index;
};

int train_dataset[6669][193];
int test_dataset[3333][193];

struct values first_fold[2223][4446];
struct values second_fold[2223][4446];
struct values third_fold[2223][4446];
struct values test[3333][6669];

int error_first_fold[25];
int error_third_fold[25];
int error_second_fold[25];

float final_error[25];

int count_mat[3333][10];

void merge(struct values arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    /* create temp arrays */
    struct values L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];


    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i].distance <= R[j].distance)
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(struct values arr[], int l, int r)
{
    if (l < r)
    {
        int m = l+(r-l)/2;
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);

        merge(arr, l, m, r);
    }
}

float calculate_distance(int ii,int j){
  float sum=0;
  int temp1[192],temp2[192];
  for(int temp=0;temp<192;temp++){
    temp1[temp] = train_dataset[ii][temp];
    temp2[temp] = train_dataset[j][temp];
    sum+=pow(temp1[temp]-temp2[temp],2);
  }

  return sqrt(sum);
}

void calculate_error(int index){
  if(index==1){
    float distance;
    #pragma omp parallel for
    for(int i=0;i<2223;i++){
      for(int j=2223;j<6669;j++){
        distance = calculate_distance(i,j);
        first_fold[i][j-2223].distance = distance;
        first_fold[i][j-2223].index = j;
      }
    }
  }
  else if(index==2){
    float distance1,distance;
    #pragma omp parallel for
    for(int i=2223;i<4446;i++){
      for(int j=0;j<2223;j++){
        distance = calculate_distance(i,j);
        distance1 = calculate_distance(i,j+4446);
        second_fold[i-2223][j].distance = distance;
        second_fold[i-2223][j].index = j;
        second_fold[i-2223][j+2223].distance = distance1;
        second_fold[i-2223][j+2223].index = j+4446;
      }
    }
  }
  else if(index==3){
    float distance;
    #pragma omp parallel for
    for(int i=6668;i>=4446;i--){
      for(int j=0;j<4446;j++){
        distance = calculate_distance(i,j);
        third_fold[6668-i][j].distance = distance;
        third_fold[6668-i][j].index = j;
      }
    }
  }
}


void calculate_error_rates(int index){
  if(index==1){
    for(int k=1;k<=25;k++){
      int error_sum=0;
      for(int i=0;i<2223;i++){
        int class = train_dataset[i][192];
        for(int j=0;j<k;j++){
          int index_c = first_fold[i][j].index;
          if(class!=train_dataset[index_c][192]){
            error_sum++;
          }
        }
      }
      int mul = 2223*k;

      error_first_fold[k-1]=error_sum;
 
    }
  }
  else if(index==2){
    for(int k=1;k<=25;k++){
      int error_sum=0;
      for(int i=2223;i<4446;i++){
        int class = train_dataset[i][192];
        for(int j=0;j<k;j++){
          int index_c = second_fold[i-2223][j].index;
          if(class!=train_dataset[index_c][192]){
            error_sum++;
          }
        }
      }
      int mul = 2223*k;

      error_second_fold[k-1]=error_sum;
    }
  }
  else if(index==3){
    for(int k=1;k<=25;k++){
      int error_sum=0;
      for(int i=6668;i>=4446;i--){
        int class = train_dataset[i][192];
        for(int j=0;j<k;j++){
          int index_c = third_fold[6668-i][j].index;
          if(class!=train_dataset[index_c][192]){
            error_sum++;
          }
        }
      }
      int mul = 2223*k;
     
      error_third_fold[k-1]=error_sum;

    }
  }
}

int min_k(){
  int max = INT_MAX;
  int index;
  for(int i=0;i<25;i++){
    if(max>final_error[i]){
      max = final_error[i];
      index = i;
    }
  }
  return index;
}

float calculate_distance_test_old(int ii,int j){
  float sum=0;
  int temp1[192],temp2[192];
  for(int temp=0;temp<192;temp++){
    temp1[temp] = train_dataset[ii][temp];
    temp2[temp] = test_dataset[j][temp];
  }
  for(int i=0;i<192;i++){
    sum+=pow(temp1[i]-temp2[i], 2);
  }
  return sqrt(sum);
}

float calculate_distance_test(int ii, int j) {
    float sum = 0;
    #pragma omp simd reduction(+:sum)
    for (int temp = 0; temp < 192; temp++) {
        float diff = train_dataset[ii][temp] - test_dataset[j][temp];
        sum += diff * diff;
    }
    return sqrt(sum);
}


void calculate_test(){
  float distance;
  
  #pragma omp parallel for collapse(2)
  for(int i=0;i<3333;i++){
    for(int j=0;j<6669;j++){
      distance = calculate_distance_test(i,j);
      test[i][j].distance = distance;
      test[i][j].index = j;
    }
  }
}

int find_max(int j){
  int min = INT_MIN;
  int index;
  for(int i=0;i<9;i++){
    if(min<count_mat[j][i]){
      min = count_mat[j][i];
      index = i;
    }
  }
  return index;
}

int main(){
    FILE *fp = fopen("pp_tra.txt","r");
    while(getc(fp)!=EOF){
      for(int i=0;i<6669;i++){
        for(int j=0;j<193;j++){
          fscanf(fp,"%d",&train_dataset[i][j]);
        }
      }
    }
    calculate_error(1);
    for(int i=0;i<2223;i++){
        mergeSort(first_fold[i],0,4446-1);
    }

    calculate_error_rates(1);

    calculate_error(2);
    for(int i=0;i<2223;i++){
      mergeSort(second_fold[i],0,4446-1);
    }


    calculate_error_rates(2);

    calculate_error(3);
    for(int i=0;i<2223;i++){
      mergeSort(third_fold[i],0,4446-1);
    }

    calculate_error_rates(3);

    printf("Final Error\n");
    for(int i=0;i<25;i++){
      float out=0;
      out = (error_first_fold[i]+error_second_fold[i]+error_third_fold[i]);
      final_error[i] = out;
      printf("%f\t",final_error[i]);
    }
    printf("\n");



    int min_index_k = min_k()+1;


    FILE *fp1 = fopen("pp_tes.txt","r");
    while(getc(fp1)!=EOF){
      for(int i=0;i<3333;i++){
        for(int j=0;j<193;j++){
          fscanf(fp1,"%d",&test_dataset[i][j]);
        }
      }
    }

    calculate_test();
    for(int i=0;i<3333;i++){
      mergeSort(test[i],0,6669-1);
    }
    for(int i=0;i<3333;i++){
      for(int j=0;j<10;j++){
        test_dataset[i][j]=0;
      }
    }

    int count=0;
    for(int i=0;i<3333;i++){
      int class = test_dataset[i][192];
      for(int j=0;j<min_index_k;j++){
        int index_c = test[i][j].index;
        int class_train = train_dataset[index_c][192];
        if(class==class_train){
          count_mat[i][class_train]+=1;
        }
      }
    }

    for(int i=0;i<3333;i++){
        int max_index = find_max(i);
        if(test_dataset[i][192]==max_index){
          count++;
        }
    }
    double p=3333-count;
    double a=(p/3333);
    double r=a*100;
    printf("r= %lf",r);
}