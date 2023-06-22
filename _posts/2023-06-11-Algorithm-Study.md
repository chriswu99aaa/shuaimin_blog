**这篇博客会记录自己学习算法是的思考以及相关学习资源**

## 数组
### 二分查找

给定一个 n 个元素**有序的**（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        // 定义左闭右闭区间
        while(left <= right)
        {
            int mid = left + ((right-left) >> 1); //不断更新寻找mid
            if(nums[mid] < target)
            {
                left = mid + 1;
            }
            else if(nums[mid] > target)
            {
                right = mid -1;
            }else{
                return mid;
            }
        }
        return -1;
    }
}
```

这个题目的细节在于

1. 对于区间的定义，循环不变量。 [left,right] 我们定义左闭右闭区间，在while 处就是<=,因为等于号是有意义的。相反[left,right)，就会是 < 因为right 不包括在区间内，所以不存在比较。
2. mid 一定是要定义在while 里面，这样就可以不断的计算mid
3. [left,right] 就是mid+1 和mid-1在对left 和right 更新时。因为前面的mid 已经包括在了区间内，我们需要再偏移一个元素。

**二分查找应用的前提是：1. 有序，2.元素不重复**

### 移除元素

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int slowIndex=0;
        for(int fastIndex=0; fastIndex<nums.length; fastIndex++)
        {
            if(nums[fastIndex] != val){
                nums[slowIndex] = nums[fastIndex];
                slowIndex++;
            }
        }
    }
}
```

使用快慢指针做两个for 循环的工作。快指针探索新的元素，慢指针决定新数组的下标。

### 有序数组平方

给你一个按**非递减顺序** 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
请你设计时间复杂度为 O(n) 的算法解决本问题

输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]

```java
class Solution {
    public int[] sortedSquares(int[] nums) {

        int[] NewArray = new int[nums.length];
        int k = nums.length -1;
        int i,j;
        for(i=0,j=nums.length-1; i<=j; )
        {
            if(nums[i]*nums[i] > nums[j]*nums[j])
            {
                NewArray[k] = nums[i]*nums[i];
                i++;
            }else{
                NewArray[k] = nums[j]*nums[j];
                j--;
            }
            k--;
        }
        return NewArray;
    }
}

``` 
这个题目利用左右双指针的思路。首先一个包涵负数的生序数组，如果求平方，就会出现中间小两边大的情况。所以利用双指针，左右扫描。这里新建一个数组，从后往前放元素。

### 长度最小的子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

```
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
```

我们是要发现一个新数组，它的元素之和大于等于target，它的长度最小。我们可以使用滑动窗口的思想。定义快慢指针，快指针不断的有一探索新的元素，慢指针不断右移获取最短子数组。慢指针移动时需要从数组和中间去慢指针位置上的数，并检查数组和是否仍然大于等于target。

**注：** 本题中使用滑动窗口主要确定一下三点

* 窗口内元素是什么？
* 如何移动窗口终止位置？
* 如何移动窗口起始位置？

```java
class Solution{
    public int minSubArrayLen(int target, int[] nums)
    {
        //
        int sum = 0;
        int result = Integer.MAX_VALUE;
        int subArrayLen = 0;
        int i=0;
        for(int j=0; j<nums.length; j++)
        {
            sum += nums[j];
            while(sum >= target)
            {
                sum -= nums[i];
                subArrayLen = j-i+1;
                result = Math.min(subArrayLen,result);
                i++;
            }
        }
        return result;
    }
}
```

### 螺旋矩阵

给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

这个问题涉及到定义循环不变量。我们定义[x,y) 左闭右开区间。还定义offset 偏移量，以及loop 这个会让我们确定有几个loop。

```java
class Solution {
    public int[][] generateMatrix(int n) {

        //loop invariant [x,y)
        int[][] result = new int[n][n];
        int startX=0;
        int startY=0;
        int offset = 1; //offset to the corner
        int loop = n/2;
        int mid = n/2;
        int count = 1; //record the number of loop run through
        int i,j;

        while(loop >= 0)
        {
            i=startX;
            j=startY;
            //left to right
            for(j=startY; j<n-offset; j++)
            {
                result[i][j] = count++;
            }
            //up to down on right
            for(i=startX; i<n-offset; i++)
            {
                result[i][j] = count++;
            }
            //right to left on down
            for(; j>startY; j--)
            {
                result[i][j] = count++;
            }
            //down to up on left
            for(; i>startX; i--)
            {
                result[i][j] = count++;
            }
            loop--;
            offset++;
            startX++;
            startY++;
        }
        if(n%2 == 1)
        {
            result[mid][mid]= count;
        }
        return result;
    }
}
```

## 链表
### 删除链表元素/ 虚拟头节点

给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
 
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]

这个题目中我们需要注意，在检查链表中元素数值前，首先要检查这个节点是否存在。要有一个零时的节点作为head 的补充来获取更后面的节点。头节点是不能改的，我们需要current指针作为遍历的指针。

```java
class Solution{
    public ListNod removeElements(ListNode head, int val)
    {
        while(head != null && head.next == val)
        {
            head = head.next;
        }
        ListNode current = head;
        while(current != null && current.next != null) //当前与下一个节点都存在。如果第二个条件不符合就代表着当前节点是最后一个
        {
            if(current.next.val == val)
            {
                current.next = current.next.next;
            }else{
                current = current.next;
            }
        }
        return head;
    }
//虚拟头节点
        public ListNod removeElements2(ListNode head, int val)
    {
        ListNode dummyhead = new ListNode(0);
        dummyhead.next = head;
        ListNode current = dummyhead;

        while(current.next != null)
        {
            if(current.next.val == val)
            {
                current.next = current.next.next;
            }else{
                current = current.next;
            }
        }
        head = dummyhead.next;
        return head;
    }
}
```
这里我们要删除的是current 的下一个元素。实际上我们删除的方式，就是把current 的指针指向下一个的下一个，这样就删除了。所以while loop 的检查是current.next 不为空。在链表的题目中，我们需要注意的是空指针的问题。

### 链表经典题目



```java
class MyLinkedList {
    int size;
    ListNode head;

    public MyLinkedList() {
        size = 0;
        head = new ListNode(0);
    }

    public int get(int index) {
        if (index < 0 || index >= size) {
            return -1;
        }
        ListNode cur = head;
        for (int i = 0; i <= index; i++) {
            cur = cur.next;
        }
        return cur.val;
    }

    public void addAtHead(int val) {
        addAtIndex(0, val);
    }

    public void addAtTail(int val) {
        addAtIndex(size, val);
    }

    public void addAtIndex(int index, int val) {
        if (index > size) {
            return;
        }
        index = Math.max(0, index);
        size++;
        ListNode pred = head;
        for (int i = 0; i < index; i++) {
            pred = pred.next;
        }
        ListNode toAdd = new ListNode(val);
        toAdd.next = pred.next;
        pred.next = toAdd;
    }

    public void deleteAtIndex(int index) {
        if (index < 0 || index >= size) {
            return;
        }
        size--;
        ListNode pred = head;
        for (int i = 0; i < index; i++) {
            pred = pred.next;
        }
        pred.next = pred.next.next;
    }
}

class ListNode {
    int val;
    ListNode next;

    public ListNode(int val) {
        this.val = val;
    }
}
```

```java
class MyLinkedList {
    int size;
    ListNode head;

    public MyLinkedList() {
        size = 0;
        head = new ListNode(0);
    }
    
    public int get(int index) {
        if(index < 0 || index > this.size-1){
            return -1;
        }
        ListNode dummyhead = new ListNode(0);

        ListNode current = dummyhead.next;
        while(index > 0 )
        {
            current = current.next;
            index--;
        }
        return current.val;
    }
    
    public void addAtHead(int val) {
        ListNode dummyhead = new ListNode(0);
        dummyhead.next = head;
        ListNode newNode = new ListNode(val);
        newNode.next = head;
        dummyhead.next = newNode;
    }
    
    public void addAtTail(int val) {
        ListNode dummyhead = new ListNode(0);
        dummyhead.next = head;
        ListNode cur = dummyhead;
        ListNode newNode = new ListNode(val);

        while(cur.next != null)
        {
            cur = cur.next;
        }
        cur.next = newNode;
        this.size++;
    }
    
    public void addAtIndex(int index, int val) {
        if(index > size)
            return;
        ListNode newNode = new ListNode(val);
        ListNode current = dummyhead;

        while(index > 0)
        {
            current = current.next;
            index--;
        }
        newNode.next = curren.next;
        current.next = newNode;
        this.size++;
    }
    
    public void deleteAtIndex(int index) {
        ListNode dummyhead = new ListNode(0);
        ListNode current = dummyhead.next;
        dummyhead.next = head;

        while(index > 0)
        {
            current = current.next;
            index--;
        }
        current.next = null;

    }
}

class ListNode{
    int val;
    ListNode next;

    public void ListNode(int val)
    {
        this.val = val;
    }
}
```

### 反转链表哦

给定单链表的头节点 head ，请反转链表，并返回反转后的链表的头节点。

输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

#### 双指针思路

双指针思路需要我们初始化当前节点，和前驱节点。pred 就是dummyhead，一开始是null，这也对应链表最后是null。
进入循环，直到current 为空。首先要存取 current.next 为 tmp，因为当我们之后对curernt.next 操作时，tmp 会
失去参考。之后就是current.next 指向 pred，current 赋给 pred，tmp 赋给 current。 最后当循环结束，让pred 成为head，因为current 当前为空指针。

```java
class Solution{
    pulic ListNode reverseList(ListNode head)
    {
        //双指针思路

        //初始化两个指针，当前指针与前驱指针
        ListNode pred = null;
        ListNode current = head;

        //当前指针是null 时，代表到达链表最后端
        while(current != null)
        {
            ListNode tmp = current.next; 
            current.next = pred; //前驱节点成为current.next
            pred = current;  //当前节点成为前驱节点
            current = tmp;  //临时节点成为
        }
        head = pred;
        return head;


        ListNode pred = null
        ListNode current = head;
        while(current != null)
        {
            ListNode tmp = current.next;
            current.next = pred;
            pred = current;
            current = tmp;
        }
        head = pred;
        return head;
    }
}
```

#### 递归法

递归法的思路其实与双指针类似，差别在于多定义了一个reverse 函数。其他的双指针初始化，都是一样的。我们通过递归的执行reverse，将链表反转。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
    ListNode pred = null;
    ListNode curr = head;
    return reverse(curr, pred);        
}

public ListNode reverse(ListNode curr, ListNode pred)
{
    if(curr == null)
        return pred;
    ListNode tmp = curr.next;
    curr.next = pred;
    return reverse(tmp, curr);
}
}
```

### 两两交换链表

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]

输入：head = []
输出：[]
```

``` java
class Solution{
    public ListNode swapParis(ListNode head)
    {
        ListNode dummyhead = new ListNode(0);
        dummyhead.next = head;
        ListNode current = dummyhead;
        ListNode firstNode, secondNode, tmp;

        while(current.next != null && current.next.next != null)
        {
            tmp = current.next.next.next;
            firstNode = current.next;
            secondNode = current.next.next;

            current.next = secondNode; //步骤一
            secondNode.next = firstNode; //步骤二
            firstNode.next = tmp //步骤三

            current = firstNode;
        }
        return dummyhead.next;
    }
}
```
为了更好的理解，我们初始化了三个变量firstNode, secondNode, tmp。循环条件，首先要写current.next 再写current.next.next 以避免空指针。因为current 是dummyhead，所以current不存在为空的现象。首先把第三个节点存储为tmp，再获取第一和第二个节点。先让current.next 指向第二个节点，再让第二个指向第一个，最后在让第一个指向第三个。此时第一个其实是第二个，因为他们互换了。这就是两两互换

### 删除链表的倒数第N个节点

给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
输入：head = [1,2], n = 1
输出：[1]
```

```java
class Solution{
    public ListNode removeNthFromEnd(ListNode head, int n)
    {
        ListNode dummhead = new ListNode(0);
        dummyhead.next = head;
        ListNode fast,slow = dummyhead;

        while(n>0 && fast !=null)
        {
            n--;
            fast = fast.next;
        }
        fast = fast.next;
        while(fast ! =null)
        {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next= slow.next.next;
        return dummyhead.next;
    }
}
```

当我们要删除第n个节点时，我们需要获取n的前一个节点。所以我们需要把快指针在第一个循环之后再移动一下。先让快指针移动n次，就是为了让快慢指针之间的距离是n，当快指针移动到末端时，慢指针刚好指向倒数第n 个节点。本体的思路是先移动快指针n+1次，然后快慢指针同时移动至末端。最后慢指针删除下一个节点。