
# C/C++ 学习笔记

# C 语言

```c
#include <stdio.h>

int main() {
   printf() displays the string inside quotation
   printf("Hello, World!");

   int price = 0;
   printf("请输入金额\n");

   scanf("%d", &price);
   int change = 100-price;

   printf("找您%d元。\n", change);

   int a,b;
   scanf("%d %d", &a, &b);

   printf("%d + %d = %d", a,b,a+b);

   const int c;
   return 0;
}

```
这段代码展示了C语言输入输出

**const 修饰词**

* const 修饰int 变量c。这个修饰词代表，变量一旦被初始化就无法被修改
* 

## C 语言文件编译过程

```c
gcc -E main.c -o main.i 预处理
gcc -S main.i -o main.s 编译
gcc -c main.s -o main.o 汇编
gcc main.o -o hello 链接

```

## C 语言整数表达

![image](../pictures/INT.png)

```c
#include <stdio.h>

int main()
{
    char c = 127;
    int i = 255;
    c = c+1;
    printf("c=%d, i=%d\n",c,i);

    return 0;
}

```

这段代码展示的就是蒸熟二进制的补码例子。char 是一个字节，八个比特。二的七次方就是128，但正数范围要少一也就是 $2^7 - 1$。第八个比特作为正负号比特来表示正负。 但如果使用unsigned 正数一个字节可表达的范围就是255。 

上图中的圆圈展示了二进制运算中一个字节正数最大范围 $127 + 1 = -128$，同理在unsigned 的标识中 $255 + 1 = 0$。 在这里二进制运算中包含模数运算的特质。也需要在计算机程序编写中意识到overflow 这个概念。

### 八进制与十六进制

一个十六进制的数可以表达四个比特的二进制数据，十六进制可以很方便的表示二进制数字

### 字符类型

在计算机系统内部，使用的ASCII 编码，数字1 和 字符‘1’ 所表达的数字是不一样的。

```c
    char c;
    char d;
    c = '1';
    d = 'A';

    if(c==d)
        printf("相等\n");
    else
        printf("不想等\n");
    
    printf("c = %d\n",c);
    printf("c = %c\n",c);
```

### 指针

```c
#include <stdio.h>

void f(int *p);
void g(int g);

int main()
{
    int i = 6;
    printf("i = %d\n",i);

// &i gets the address of the variable i
    f(&i);

// the value of i is changed to 26 by pointer
    g(i);
    return 0;
}

void f(int *p)
{
    printf("p = %p\n",p);
    printf("p = %d\n", *p);
    *p = 26;
}

void g(int k)
{
    printf("k = %d\n",k);
}
```


* \*p 代表指针p 指向位置的value
* &p 代表p的位置

我们可以看到 * 和 & 是两个可逆的操作，一个获取变量的地址，一个获取这个地址上的值。 

在C 语言中，当我们在函数参数中传一个数组，那实际上是传递了数组第0个元素的地址。在计算机操作系统中缓存可以想象成一个大数组，在一个数组中，根据数据类型比如int，四个字节，每四个字节存一个元素。如果数组中有八个元素，那么这个数组就会有32个字节。

### const 修饰词

const 要注意在 *  前还是后
1. const 在 * 前， 代表const 所修饰的那个东西不能变
2. const 在 * 后， 代表指针不能变。

```c
int i;
const int *p1 = &i;
int const* p2 = &i;
int *const p3 = &i;
```

在上面的例子中第一二个例子是相等的，而第三个就代表指针是 const。

#### const 数组

```c
const int a[] = {1,2,3,4,5,6};
```

在这个例子中，数组中所有的元素就是const，所以是不能修改的，所以只能使用初始化赋值


## C++ 面向对象编程

### head文件

```cpp

complex.h

#ifndef __COMPLEX__
#define __COMPLEX__

class declaration
class complex 
{
	public:
		complex (double r=0, double i=0)
		: re(r), im(i)
		{}
		/* : re(r), im(i) 是运用了cpp编译原理更高效 */

		complex& operator += (const complex&);

		double real() const {return re;}

		double imag() const {return im;}
	private:
		T re, im;

	/* T 就是模版. Generic type*/
		
	friend complex& __doapl (complex*, const complex&)



};




class definition
complex::function
#endif
```


**Pass by Reference: complex&. (Here & is the reference symbol)**


**Pass by Value: complex**

**In general, use Pass by Reference is better**





