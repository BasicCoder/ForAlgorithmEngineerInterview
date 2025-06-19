# 思谋科技

## 一面：

1.算法题：一个数组包含不同的数字，查找随机一个峰值数字(二分)
```C++
class Solution{
public:
    int findLocalMax(vector<int> & nums){
        if (nums[0] > nums[1]) {
            return nums[0];
        }
        int n = nums.size();
        if (nums[n-1] > nums[n-2]){
            return nums[n-1];
        }
        int start = 1, end = n -2;
        while(start <= end){
            int mid = (start + end) / 2;
            if (nums[mid -1] < nums[mid] && nums[mid] > nums[mid +1]) {
                return nums[mid];
            } else if (nums[mid - 1] < nums[mid]){
                start = mid + 1;
            } else {
                end = mid - 1;
            }

        }
        return -1;
    }
};
```

## 二面：

1.算法题：给定一个数字，输出这个数字不同的位数重新排列后，数值最小的数字。
```C++

```

## 三面：
项目相关知识点

--------------------------
# 地平线
## 一面：

1.算法题：有序数组中找到目标数字所在的区间位置(二分)
```C++
class Solution{
public:
    vector<int> findLocation(vector<int> &nums, int target){
        vector<int> res = {-1, -1};
        int i = 0;
        int j = nums.size();
        int start = -1;
        int end = -1;
        while(i < j){
            int mid = (i + j) / 2;
            if (nums[mid] == target){
                start = mid;
                j = mid;
            }else if (nums[mid] < target){
                i = mid + 1;
            }else if (nums[mid] > target){
                j = mid;
            }
        }

        i = 0;
        j = nums.size();
        while(i < j){
            int mid = (i + j) / 2;
            if (nums[mid] == target){
                end = mid;
                i = mid + 1;
            }else if (nums[mid] < target){
                i = mid + 1;
            }else if (nums[mid] > target){
                j = mid;
            }
        }

        res[0] = start;
        res[1] = end;
        return res;
    }
};
```
-------------------------
# 格林深瞳
## 一面：

-------------------------
# B站
## 岗位：
高级研发工程师
### 一面：
1.算法题：
有n个人和m对朋友关系，假设朋友关系具有传递性，问n个人共有多少个朋友圈。
采用并查集
```C++
class Solution{
public:
    int get_root(int x, vector<int>& root){
        if (root[x] == x){
            return x;
        } else {
            root[x] = get_root(root[x], root);
            return root[x];
        }
    }
    int func(vector< vector<int> > &inputs, int n, int m){
        vector<int> root(n +1, -1);
        for(int i = 1; i <=n; ++i){
            root[i] = i;
        }
        
        for(int i = 1; i <=n; ++i){
            for(int j: inputs[i]) {
                int root_i = get_root(i, root);
                int root_j = get_root(j, root);
                if (root_i != root_j){
                    root[root_j] = root_i;
                }
            }
        }
        vector<int> unique_root(n +1);
        for(int i = 1; i <= n; ++i){
            int r = get_root(i, root);
            unique_root[r]= 1;
        }
        int res = 0;
        for(int i: unique_root) {
            res += i;
        }
        return res;
    }
};
```
### 二面：
1.算法题(二选一)：

数据流中的中位数
```C++
class Solution{
  pubulic:
      void addNum(int num){
          max_heap.push(num);
          min_heap.push(max_heap.top());
          max_heap.pop();
          if (max_heap.size() < min_heap.size()){
              max_heap.push(min_heap.top());
              min_heap.pop();
          }
      }
      double MedianFinder(){
          return max_heap.size() > min_heap.size() ? max_heap.top() : (max_heap.top() + min_heap.top()) / 2
      }
  
  private:
      priority_queue<int> max_heap;
      priority_queue<int, vector<int>, greater<int> > min_heap;
      
  };
```

N皇后问题

### 三面：
1.算法题:
判断一个单向链表是否存在环，若存在，输出入口节点位置。
```C++
struct Node{
    int val;
    Node* next;
    Node(int _val):val(_val){}
};
class Solution{
public:
    bool isRing(Node* head){
        if (!head) return false;
        Node* slow = head;
        Node* fast = head;
        do{
            if (!fast -> next || !fast -> next -> next){
                return false;
            }
            slow = slow -> next;
            fast = fast -> next -> next;
        }while(slow != fast);
        return true;
    }
    Node* findEnter(Node* head){
        if (!head) return NULL;
        Node *p, *q;
        q = head, p = head -> next;
        if( !p) return NULL;
        while(p &&q && p != q){
            p = p -> next;
            q = q -> next;
            if (p){
                p = p -> next;
            }else{
                return NULL;
            }
        }
        p = head, q = q -> next;
        while(p != q){
            p = p -> next;
            q = q -> next;
        }
        return p;
    }
};
```

-------------------------
# 小红书
## 岗位：
多模态资深算法工程师
### 一面：
1.算法题：矩阵顺时针旋转90度，两种情况：N x N 方阵旋转和 N x M 矩阵旋转
```C++
class Solution{
public:
    void rotate(vector< vector<int> > & matrix){
        int row = matrix.size();
        int n = row;
        i,j =  j, n - i +1 = n - i +1, n -j +1, n -j +
        for(int i = 0; i < n /2; ++i){
            for(int j = i; j < n - i - 1; ++j) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n -j -1][i];
                matrix[n -j - 1][i] = matrix[n -i - 1][n -j -1];
                matrix[n -i - 1][n - j - 1] = matrix[j][n -i - 1];
                matrix[j][n - i -1] = tmp;
            }
        }
    }
    void rotate1(vector< vector<int> > & matrix){
        int row = matrix.size();
        int n = row;
        for (int i = 0; i < n; ++i){
            for (int j = i + 1; j < n; ++j){
                swap(matrix[i][j], matrix[j][i]);
            }
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
};
```
### 二面：
1.算法题：n x m 矩阵， 输入每个请求：左上角和右下角的坐标，输出：子矩阵和 
```C++
class Solution{
public:
    void computer_prefix_sum(vector< vector<int> >& m){
        int rows = m.size();
        int cols = m[0].size();
        for (int r = 0; r < rows; ++r){
            for(int c = 0; c < cols; ++c){
                if (c > 0) m[r][c] += m[r][c - 1];
                if (r > 0) m[r][c] += m[r-1][c];
                if (r > 0 && c > 0) m[r][c] -= m[r-1][c-1];
            }
        }
    }
    int findSum(vector< vector<int> >& m, int r1, int c1, int r2, int c2){
        //int rows = matrix.size();
        //int cols = matrix[0].size();
        computer_prefix_sum(m);
        return m[r2][c2] - 
            (c1 > 0 ? m[r2][c1 -1]:0) -
            (r1 > 0 ? m[r1 - 1][c2]:0) +
            (r1 > 0 && c1 > 0 ? m[r1-1][c1-1]:0);
    }
};
```
-------------------------
# 美团
## 岗位：
移动端推理算法优化专家
### 一面：
1.算法题：合并两个二叉搜索树
2.算法题：有序数组，还原成平衡二叉树
```C++
struct Node{
    int val;
    Node* left;
    Node* right;
    Node(int val_, Node* left_ = NULL, Node* right_= NULL){
        val = val_;
        left = left_;
        right = right_;
    }
};
Node* dfs(int arr[], int left, int right){
    if (left < right){
        return NULL;
    }
    int mid = (left + right) / 2;
    Node* root = new Node(arr[mid]);
    root -> left = dfs(arr, left, mid -1);
    root -> right = dfs(arr, mid + 1, right);
    return root;
}
Node* build_tree(int arr[], int size){
    return dfs(arr, 0, size -1);
}
```

### 三面：

1.算法题：用Neon指令切分RGB的三个通道
```C++
class Solution{
    void splitRGB(void *src, int len, void *dst){
        int num8x16 = len / 16;
        uint8x16x3_t vdst;
        for (int i = 0; i < num8x16; ++i)
            vdst = vld3q_u8(src+ 3 * 16 * i);
            vst1q_u8(dst + 16 * i, vdst.val[0]); //R
            vst1q_u8(dst + len / 3 + 16 * i, vdst.val[1]); //G
            vst1q_u8(dst + len / 3 * 2 + 16 * i, vdst.val[2]); //B
        }
};
```

-------------------------
# PDD (不匹配，结束)
## 岗位：
搜索算法-图像方向
### 一面：
1.场景问题：多目标优化、联合优化问题

### 二面：
2.算法题：
两个有序数组中位数

# 字节 （不匹配）
## 岗位：
计算机视觉算法工程师
### 一面：
1.算法题: 跳跃游戏LeetCode45

给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置
可以跳跃到的最大长度。你的目标是使用最少的跳跃次数达到数组的最后一个位置。

```C++
class Solution{
public:
    int jump(vector<int> &nums){
        int i = 0, j = 1, steps = 0, n = nums.size();
        while(j < n){
            int end = min(nums[i] + i + 1, end);
            while(j < end){
                if (nums[j] + j > nums[i] + i){
                    i = j;
                }
                ++j;
            }
            ++steps;
        }
        return steps;
    }
};
```
# Shopee
## 岗位：
图像处理算法工程师
### 一面：
1.算法题: 求三个矩形ROI的交集区域

```C++
int area(int x1, int y1, int x2, int y2){
    return (x2 - x1) * (y2 - y1);
}
vector<int> get_intersection(int left1, int top1, int right1, int bottom1, int left2, int top2, int right2, int bottom2){
    int left3 = max(left1, left2);
    int right3 = min(right1, right2);
    int top3 = max(top1, top2);
    int bottom3 = min(bottom1, bottom2);
    
    if(left3 > right3 || top3 > bottom3){
        return {-1, -1, -1, -1};
    } else {
        return {left3, right3, top3, bottom3};
    }
}
int get_insertROI(vector<int> &a, vector<int> &b, vector<int> &c){
    vector<int> ab = get_intersection(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
    vector<int> ac = get_intersection(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3]);
    vector<int> bc = get_intersection(b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3]);

    vector<int> abc = get_intersection(ab[0],ab[1], ab[2], ab[3], c[0], c[1], c[2], c[3]);

    int sum1 = aera(ab[0], ab[1], ab[2], ab[3]);
    int sum2 = aera(ac[0], ac[1], ac[2], ac[3]);
    int sum3 = aera(bc[0], bc[1], bc[2], bc[3]);

    int sum = sum1 + sum2 + sum3 - 2 * area(abc[0], abc[1], abc[2], abc[3]);

    return sum;
}
```

2.算法题：求最大的矩形面积

```C++
class Solution{
public:
    int largestRectangleArea(vector<int> & heights){
        int n = heights.size();
        vector<int> left(n), right(n);
        stack<int> mono_stack;
        
        for(int i = 0; i < n; ++i){
            while(!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]){
                mono_stack.pop();
            }
            left[i] = (mono_stack.empty() ? -1: mono_stack.top());
            mono_stack.push(i);
        }

        mono_stack = stack<int>();

        for(int i = n -1; i >= 0; --i){
            while(!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]){
                mono_stack.pop();
            }
            right[i] = (mono_stack.empty() ? n: mono_stack.top());
            mono_stack.push(i);
        }

        int ans = 0;
        for(int i = 0; i < n; ++i){
            ans = max(ans, (right[i] - left[i] - 2) * heights[i]);
        }
        return ans;
    }
};
```
