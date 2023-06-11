## Binary Search

It's not neccessary to apply binary search in sorted array, when the is about the left and right
side of an element, and one side can be drop off, then we can consider binary search

## Sorting

只要是不基于比较的排序，都需要根据数据状况来进行排序

## Heap

``` java
public static void heapInsert(int[] arr, int index)
{
    while(arr[index] < arr[(index-1) / 2])
    {
        heap.swap(arr,index, (index-1)/2);
        index = (index-1) / 2;
    }
}

public static void heapify(int[] arr, int index, int heapSize)
{
    int left = 2*index + 1;
    while(left < heapSize)
    {
        int largest = left+1 < heapSize && arr[left+1] > arr[left] ? left+1 : left;

        largest = arr[index] > arr[largest] ? index : largest;

        if(largest == index)
            break;
        heap.swap(arr, index, largest);
        index = largest; // the curent index changes to the postion of the largest after swap.
        left = 2 * index + 1;
    }
}

public static void heapSort(int[] arr, int index, int heapSize)
{
    if(arr == null || arr.length < 2)
        return;

    for(int i=0; i<arr.length; i++)
    {
        heapInsert(arr,i);
    }

    heap.swap(arr,0,--heapSize);
    while(heapSize > 0)
    {
        heapify(arr,0, heapSize);
        heap.swap(arr, 0, --heapSize);
    }
}
```