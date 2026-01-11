# 期货量化系统前端重构工程计划

**项目名称**：期货回测系统 UI 现代化重构
**目标**：从 Streamlit 迁移到 React + FastAPI 架构
**预计工时**：15-20天
**设计风格**：浅色调（米白底色 + 紫色强调色）

---

## 一、项目概览

### 1.1 现状
- **前端**：Streamlit (Python)
- **样式**：自定义CSS + HTML
- **图表**：Plotly
- **状态管理**：st.session_state

### 1.2 目标
- **前端**：React 19 + Vite + TypeScript
- **样式**：Tailwind CSS + shadcn/ui
- **图表**：Lightweight Charts (K线) + Recharts (统计图)
- **后端API**：FastAPI
- **设计风格**：浅色米白底 + 紫色强调色

### 1.3 目录结构（最终）

```
D:\期货\回测改造\
├── backend/                    # FastAPI 后端
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── market.py           # 行情数据 API
│   │   ├── backtest.py         # 回测 API
│   │   ├── trading.py          # 交易 API
│   │   ├── settings.py         # 设置 API
│   │   └── optimizer.py        # 优化器 API
│   └── schemas/
│       ├── __init__.py
│       └── models.py           # Pydantic 数据模型
│
├── frontend/                   # React 前端
│   ├── public/
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/             # shadcn/ui 组件
│   │   │   ├── charts/         # 图表组件
│   │   │   ├── layout/         # 布局组件
│   │   │   └── shared/         # 共享组件
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Backtest.tsx
│   │   │   ├── LiveTrading.tsx
│   │   │   ├── SimTrading.tsx
│   │   │   ├── RiskCenter.tsx
│   │   │   ├── Optimizer.tsx
│   │   │   └── Settings.tsx
│   │   ├── services/           # API 调用
│   │   ├── hooks/              # 自定义 Hooks
│   │   ├── lib/                # 工具函数
│   │   ├── types/              # TypeScript 类型
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── core/                       # 保持不变 - 核心业务逻辑
├── strategies/                 # 保持不变 - 策略代码
├── data/                       # 保持不变 - 数据存储
├── configs/                    # 保持不变 - 配置文件
└── requirements.txt            # 更新 - 添加 FastAPI 依赖
```

---

## 二、设计规范

### 2.1 色彩系统

```css
/* 主色调 - 浅色米白系 */
--background: #FAFAF8;          /* 米白背景 */
--foreground: #1A1A1A;          /* 深灰文字 */

/* 卡片和容器 */
--card: #FFFFFF;                /* 纯白卡片 */
--card-foreground: #1A1A1A;

/* 强调色 - 紫色系 */
--primary: #7C3AED;             /* 主紫色 */
--primary-light: #A78BFA;       /* 浅紫色 */
--primary-dark: #5B21B6;        /* 深紫色 */
--primary-foreground: #FFFFFF;

/* 辅助色 */
--secondary: #F3F4F6;           /* 浅灰背景 */
--secondary-foreground: #374151;

/* 边框 */
--border: #E5E7EB;              /* 浅灰边框 */
--border-accent: #7C3AED;       /* 紫色边框（强调） */

/* 功能色 */
--success: #10B981;             /* 绿色 - 盈利/成功 */
--danger: #EF4444;              /* 红色 - 亏损/错误 */
--warning: #F59E0B;             /* 橙色 - 警告 */
--info: #3B82F6;                /* 蓝色 - 信息 */

/* 交易色（中国习惯：红涨绿跌） */
--profit: #EF4444;              /* 红色 - 盈利 */
--loss: #10B981;                /* 绿色 - 亏损 */

/* 图表色 */
--chart-up: #EF4444;            /* K线涨 */
--chart-down: #10B981;          /* K线跌 */
--chart-ma5: #7C3AED;           /* MA5 紫色 */
--chart-ma10: #F59E0B;          /* MA10 橙色 */
--chart-ma20: #3B82F6;          /* MA20 蓝色 */
```

### 2.2 字体系统

```css
/* 主字体 */
--font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

/* 代码/数字字体 */
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;

/* 字号 */
--text-xs: 12px;
--text-sm: 14px;
--text-base: 16px;
--text-lg: 18px;
--text-xl: 20px;
--text-2xl: 24px;
--text-3xl: 30px;
```

### 2.3 间距系统

```css
/* 基于 4px 网格 */
--spacing-1: 4px;
--spacing-2: 8px;
--spacing-3: 12px;
--spacing-4: 16px;
--spacing-5: 20px;
--spacing-6: 24px;
--spacing-8: 32px;
--spacing-10: 40px;
--spacing-12: 48px;
```

### 2.4 阴影系统

```css
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07);
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
```

### 2.5 圆角

```css
--radius-sm: 4px;
--radius-md: 8px;
--radius-lg: 12px;
--radius-xl: 16px;
```

---

## 三、分阶段执行计划

### Phase 1：项目初始化（第1-2天）

#### 任务 1.1：创建前端项目

```bash
cd "D:\期货\回测改造"
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

#### 任务 1.2：安装依赖

```bash
# UI 相关
npm install tailwindcss postcss autoprefixer
npm install class-variance-authority clsx tailwind-merge
npm install lucide-react
npm install @radix-ui/react-dialog @radix-ui/react-tabs @radix-ui/react-select @radix-ui/react-dropdown-menu @radix-ui/react-tooltip @radix-ui/react-accordion

# 图表
npm install lightweight-charts recharts

# HTTP 客户端
npm install axios

# 字体
npm install @fontsource/inter @fontsource/jetbrains-mono

# 动画（可选）
npm install framer-motion

# 开发依赖
npm install -D @types/node
```

#### 任务 1.3：配置 Tailwind

创建 `frontend/tailwind.config.js`：

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#FAFAF8",
        foreground: "#1A1A1A",
        card: {
          DEFAULT: "#FFFFFF",
          foreground: "#1A1A1A",
        },
        primary: {
          DEFAULT: "#7C3AED",
          light: "#A78BFA",
          dark: "#5B21B6",
          foreground: "#FFFFFF",
        },
        secondary: {
          DEFAULT: "#F3F4F6",
          foreground: "#374151",
        },
        border: "#E5E7EB",
        muted: {
          DEFAULT: "#F3F4F6",
          foreground: "#6B7280",
        },
        accent: {
          DEFAULT: "#7C3AED",
          foreground: "#FFFFFF",
        },
        success: "#10B981",
        danger: "#EF4444",
        warning: "#F59E0B",
        info: "#3B82F6",
        profit: "#EF4444",
        loss: "#10B981",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        lg: "12px",
        md: "8px",
        sm: "4px",
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'card-hover': '0 4px 6px rgba(0, 0, 0, 0.07)',
      },
    },
  },
  plugins: [],
}
```

#### 任务 1.4：配置全局样式

创建 `frontend/src/index.css`：

```css
@import '@fontsource/inter/400.css';
@import '@fontsource/inter/500.css';
@import '@fontsource/inter/600.css';
@import '@fontsource/inter/700.css';
@import '@fontsource/jetbrains-mono/400.css';
@import '@fontsource/jetbrains-mono/500.css';

@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 45 20% 98%;
    --foreground: 0 0% 10%;
    --card: 0 0% 100%;
    --card-foreground: 0 0% 10%;
    --primary: 263 70% 58%;
    --primary-foreground: 0 0% 100%;
    --secondary: 220 14% 96%;
    --secondary-foreground: 220 9% 46%;
    --muted: 220 14% 96%;
    --muted-foreground: 220 9% 46%;
    --accent: 263 70% 58%;
    --accent-foreground: 0 0% 100%;
    --border: 220 13% 91%;
    --input: 220 13% 91%;
    --ring: 263 70% 58%;
    --radius: 8px;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground font-sans antialiased;
  }
}

/* 自定义滚动条 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: #F3F4F6;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb {
  background: #D1D5DB;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: #9CA3AF;
}

/* 数字等宽 */
.tabular-nums {
  font-variant-numeric: tabular-nums;
}
```

#### 任务 1.5：配置 Vite

更新 `frontend/vite.config.ts`：

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8100',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8100',
        ws: true,
      }
    }
  }
})
```

#### 任务 1.6：创建后端入口

创建 `backend/main.py`：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="期货量化系统 API",
    description="支持回测、交易、数据管理的后端 API",
    version="2.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "期货量化系统 API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 导入并注册路由（后续任务创建）
# from backend.routers import market, backtest, trading, settings, optimizer
# app.include_router(market.router, prefix="/api/market", tags=["行情数据"])
# app.include_router(backtest.router, prefix="/api/backtest", tags=["回测"])
# app.include_router(trading.router, prefix="/api/trading", tags=["交易"])
# app.include_router(settings.router, prefix="/api/settings", tags=["设置"])
# app.include_router(optimizer.router, prefix="/api/optimizer", tags=["优化器"])
```

#### 任务 1.7：更新 requirements.txt

在 `requirements.txt` 中添加：

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

---

### Phase 2：基础组件库（第3-5天）

#### 任务 2.1：创建目录结构

```bash
mkdir -p frontend/src/components/ui
mkdir -p frontend/src/components/charts
mkdir -p frontend/src/components/layout
mkdir -p frontend/src/components/shared
mkdir -p frontend/src/services
mkdir -p frontend/src/hooks
mkdir -p frontend/src/lib
mkdir -p frontend/src/types
mkdir -p frontend/src/pages
```

#### 任务 2.2：工具函数

创建 `frontend/src/lib/utils.ts`：

```typescript
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// 格式化数字
export function formatNumber(value: number, decimals: number = 2): string {
  return value.toLocaleString('zh-CN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

// 格式化百分比
export function formatPercent(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`
}

// 格式化金额
export function formatCurrency(value: number, currency: string = 'CNY'): string {
  return value.toLocaleString('zh-CN', {
    style: 'currency',
    currency,
  })
}

// 格式化日期
export function formatDate(date: string | Date, format: 'date' | 'datetime' = 'date'): string {
  const d = typeof date === 'string' ? new Date(date) : date
  if (format === 'datetime') {
    return d.toLocaleString('zh-CN')
  }
  return d.toLocaleDateString('zh-CN')
}
```

#### 任务 2.3：Button 组件

创建 `frontend/src/components/ui/button.tsx`：

```tsx
import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary-dark",
        destructive: "bg-danger text-white hover:bg-danger/90",
        outline: "border border-border bg-transparent hover:bg-secondary",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-secondary hover:text-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        success: "bg-success text-white hover:bg-success/90",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-12 rounded-md px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, children, disabled, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg className="mr-2 h-4 w-4 animate-spin" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        )}
        {children}
      </button>
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
```

#### 任务 2.4：Card 组件

创建 `frontend/src/components/ui/card.tsx`：

```tsx
import * as React from "react"
import { cn } from "@/lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-lg border border-border bg-card text-card-foreground shadow-card",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn("text-lg font-semibold leading-none tracking-tight", className)}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }
```

#### 任务 2.5：MetricCard 组件（业务组件）

创建 `frontend/src/components/shared/MetricCard.tsx`：

```tsx
import React from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { cn, formatNumber, formatPercent } from '@/lib/utils'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: number | string
  subtitle?: string
  change?: number
  changeLabel?: string
  format?: 'number' | 'percent' | 'currency' | 'none'
  decimals?: number
  icon?: React.ReactNode
  className?: string
  valueClassName?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  change,
  changeLabel,
  format = 'number',
  decimals = 2,
  icon,
  className,
  valueClassName,
}) => {
  const formatValue = (val: number | string) => {
    if (typeof val === 'string') return val
    switch (format) {
      case 'percent':
        return formatPercent(val, decimals)
      case 'currency':
        return `¥${formatNumber(val, decimals)}`
      case 'number':
        return formatNumber(val, decimals)
      default:
        return String(val)
    }
  }

  const getChangeIcon = () => {
    if (change === undefined || change === 0) return <Minus className="h-4 w-4" />
    return change > 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />
  }

  const getChangeColor = () => {
    if (change === undefined || change === 0) return 'text-muted-foreground'
    return change > 0 ? 'text-profit' : 'text-loss'
  }

  return (
    <Card className={cn("hover:shadow-card-hover transition-shadow", className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          {icon && <div className="text-muted-foreground">{icon}</div>}
        </div>
        <div className="mt-2">
          <p className={cn("text-2xl font-bold font-mono tabular-nums", valueClassName)}>
            {formatValue(value)}
          </p>
          {(subtitle || change !== undefined) && (
            <div className="mt-1 flex items-center gap-2">
              {change !== undefined && (
                <span className={cn("flex items-center gap-1 text-sm", getChangeColor())}>
                  {getChangeIcon()}
                  {formatPercent(Math.abs(change))}
                </span>
              )}
              {changeLabel && (
                <span className="text-sm text-muted-foreground">{changeLabel}</span>
              )}
              {subtitle && !change && (
                <span className="text-sm text-muted-foreground">{subtitle}</span>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
```

#### 任务 2.6：Tabs 组件

创建 `frontend/src/components/ui/tabs.tsx`：

```tsx
import * as React from "react"
import * as TabsPrimitive from "@radix-ui/react-tabs"
import { cn } from "@/lib/utils"

const Tabs = TabsPrimitive.Root

const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-secondary p-1 text-muted-foreground",
      className
    )}
    {...props}
  />
))
TabsList.displayName = TabsPrimitive.List.displayName

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-sm",
      className
    )}
    {...props}
  />
))
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
      className
    )}
    {...props}
  />
))
TabsContent.displayName = TabsPrimitive.Content.displayName

export { Tabs, TabsList, TabsTrigger, TabsContent }
```

#### 任务 2.7：Select 组件

创建 `frontend/src/components/ui/select.tsx`：

```tsx
import * as React from "react"
import * as SelectPrimitive from "@radix-ui/react-select"
import { Check, ChevronDown, ChevronUp } from "lucide-react"
import { cn } from "@/lib/utils"

const Select = SelectPrimitive.Root
const SelectGroup = SelectPrimitive.Group
const SelectValue = SelectPrimitive.Value

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-border bg-card px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
))
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border border-border bg-card text-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
))
SelectContent.displayName = SelectPrimitive.Content.displayName

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-secondary focus:text-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
))
SelectItem.displayName = SelectPrimitive.Item.displayName

export {
  Select,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectItem,
}
```

#### 任务 2.8：Input 组件

创建 `frontend/src/components/ui/input.tsx`：

```tsx
import * as React from "react"
import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-border bg-card px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
```

#### 任务 2.9：Badge 组件

创建 `frontend/src/components/ui/badge.tsx`：

```tsx
import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        success: "border-transparent bg-success text-white",
        danger: "border-transparent bg-danger text-white",
        warning: "border-transparent bg-warning text-white",
        outline: "text-foreground border-border",
        profit: "border-transparent bg-profit/10 text-profit",
        loss: "border-transparent bg-loss/10 text-loss",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
```

#### 任务 2.10：Table 组件

创建 `frontend/src/components/ui/table.tsx`：

```tsx
import * as React from "react"
import { cn } from "@/lib/utils"

const Table = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement>
>(({ className, ...props }, ref) => (
  <div className="relative w-full overflow-auto">
    <table
      ref={ref}
      className={cn("w-full caption-bottom text-sm", className)}
      {...props}
    />
  </div>
))
Table.displayName = "Table"

const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("[&_tr]:border-b", className)} {...props} />
))
TableHeader.displayName = "TableHeader"

const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("[&_tr:last-child]:border-0", className)}
    {...props}
  />
))
TableBody.displayName = "TableBody"

const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "border-b border-border transition-colors hover:bg-secondary/50 data-[state=selected]:bg-secondary",
      className
    )}
    {...props}
  />
))
TableRow.displayName = "TableRow"

const TableHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0",
      className
    )}
    {...props}
  />
))
TableHead.displayName = "TableHead"

const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td
    ref={ref}
    className={cn("p-4 align-middle [&:has([role=checkbox])]:pr-0", className)}
    {...props}
  />
))
TableCell.displayName = "TableCell"

export {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
}
```

#### 任务 2.11：组件索引文件

创建 `frontend/src/components/ui/index.ts`：

```typescript
export * from './button'
export * from './card'
export * from './tabs'
export * from './select'
export * from './input'
export * from './badge'
export * from './table'
```

---

### Phase 3：K线图表组件（第6-7天）

#### 任务 3.1：K线图组件

创建 `frontend/src/components/charts/CandlestickChart.tsx`：

```tsx
import React, { useEffect, useRef, useCallback } from 'react'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  UTCTimestamp,
  ColorType,
  CrosshairMode,
} from 'lightweight-charts'

export interface KlineData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface TradeMarker {
  time: string
  position: 'aboveBar' | 'belowBar'
  color: string
  shape: 'arrowUp' | 'arrowDown' | 'circle'
  text: string
}

interface CandlestickChartProps {
  data: KlineData[]
  markers?: TradeMarker[]
  height?: number
  showVolume?: boolean
  showMA?: boolean
  maperiods?: number[]
  onCrosshairMove?: (price: number | null, time: string | null) => void
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  markers = [],
  height = 400,
  showVolume = true,
  showMA = true,
  maperiods = [5, 10, 20],
  onCrosshairMove,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // 计算MA
  const calculateMA = useCallback((data: KlineData[], period: number) => {
    const result: { time: UTCTimestamp; value: number }[] = []
    for (let i = period - 1; i < data.length; i++) {
      let sum = 0
      for (let j = 0; j < period; j++) {
        sum += data[i - j].close
      }
      result.push({
        time: (new Date(data[i].time).getTime() / 1000) as UTCTimestamp,
        value: sum / period,
      })
    }
    return result
  }, [])

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return

    // 创建图表
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#6B7280',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#E5E7EB' },
        horzLines: { color: '#E5E7EB' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#7C3AED',
          width: 1,
          style: 2,
          labelBackgroundColor: '#7C3AED',
        },
        horzLine: {
          color: '#7C3AED',
          width: 1,
          style: 2,
          labelBackgroundColor: '#7C3AED',
        },
      },
      rightPriceScale: {
        borderColor: '#E5E7EB',
      },
      timeScale: {
        borderColor: '#E5E7EB',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    chartRef.current = chart

    // K线系列
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#EF4444',
      downColor: '#10B981',
      borderUpColor: '#EF4444',
      borderDownColor: '#10B981',
      wickUpColor: '#EF4444',
      wickDownColor: '#10B981',
    })
    candleSeriesRef.current = candleSeries

    // 转换数据格式
    const chartData: CandlestickData[] = data.map((d) => ({
      time: (new Date(d.time).getTime() / 1000) as UTCTimestamp,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))

    candleSeries.setData(chartData)

    // 添加交易标记
    if (markers.length > 0) {
      const formattedMarkers = markers.map((m) => ({
        time: (new Date(m.time).getTime() / 1000) as UTCTimestamp,
        position: m.position,
        color: m.color,
        shape: m.shape,
        text: m.text,
      }))
      candleSeries.setMarkers(formattedMarkers as any)
    }

    // MA 均线
    if (showMA) {
      const maColors = ['#7C3AED', '#F59E0B', '#3B82F6']
      maperiods.forEach((period, index) => {
        const maData = calculateMA(data, period)
        const maSeries = chart.addLineSeries({
          color: maColors[index % maColors.length],
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        maSeries.setData(maData)
      })
    }

    // 成交量
    if (showVolume && data.some((d) => d.volume !== undefined)) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#7C3AED',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })

      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })

      const volumeData = data.map((d) => ({
        time: (new Date(d.time).getTime() / 1000) as UTCTimestamp,
        value: d.volume || 0,
        color: d.close >= d.open ? 'rgba(239, 68, 68, 0.5)' : 'rgba(16, 185, 129, 0.5)',
      }))
      volumeSeries.setData(volumeData)
    }

    // 十字线移动事件
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove((param) => {
        if (param.time && param.seriesData.get(candleSeries)) {
          const data = param.seriesData.get(candleSeries) as CandlestickData
          onCrosshairMove(data.close, param.time.toString())
        } else {
          onCrosshairMove(null, null)
        }
      })
    }

    chart.timeScale().fitContent()

    // 响应式
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data, markers, height, showVolume, showMA, maperiods, calculateMA, onCrosshairMove])

  return (
    <div ref={chartContainerRef} className="w-full" style={{ height }} />
  )
}

export default CandlestickChart
```

#### 任务 3.2：资金曲线组件

创建 `frontend/src/components/charts/EquityChart.tsx`：

```tsx
import React from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { formatNumber, formatPercent } from '@/lib/utils'

interface EquityData {
  date: string
  equity: number
  drawdown?: number
}

interface EquityChartProps {
  data: EquityData[]
  height?: number
  showDrawdown?: boolean
  initialCapital?: number
}

export const EquityChart: React.FC<EquityChartProps> = ({
  data,
  height = 300,
  showDrawdown = true,
  initialCapital = 1000000,
}) => {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="rounded-lg border border-border bg-card p-3 shadow-md">
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-sm font-medium">
            净值: <span className="font-mono">¥{formatNumber(payload[0].value)}</span>
          </p>
          {payload[1] && (
            <p className="text-sm font-medium text-danger">
              回撤: <span className="font-mono">{formatPercent(payload[1].value)}</span>
            </p>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#7C3AED" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#7C3AED" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 12, fill: '#6B7280' }}
          tickLine={{ stroke: '#E5E7EB' }}
          axisLine={{ stroke: '#E5E7EB' }}
        />
        <YAxis
          yAxisId="equity"
          tick={{ fontSize: 12, fill: '#6B7280' }}
          tickLine={{ stroke: '#E5E7EB' }}
          axisLine={{ stroke: '#E5E7EB' }}
          tickFormatter={(value) => `${(value / 10000).toFixed(0)}万`}
        />
        {showDrawdown && (
          <YAxis
            yAxisId="drawdown"
            orientation="right"
            tick={{ fontSize: 12, fill: '#6B7280' }}
            tickLine={{ stroke: '#E5E7EB' }}
            axisLine={{ stroke: '#E5E7EB' }}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
        )}
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine
          yAxisId="equity"
          y={initialCapital}
          stroke="#9CA3AF"
          strokeDasharray="3 3"
          label={{ value: '初始资金', fill: '#9CA3AF', fontSize: 12 }}
        />
        <Area
          yAxisId="equity"
          type="monotone"
          dataKey="equity"
          stroke="#7C3AED"
          strokeWidth={2}
          fillOpacity={1}
          fill="url(#colorEquity)"
        />
        {showDrawdown && (
          <Area
            yAxisId="drawdown"
            type="monotone"
            dataKey="drawdown"
            stroke="#EF4444"
            strokeWidth={1}
            fillOpacity={1}
            fill="url(#colorDrawdown)"
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default EquityChart
```

---

### Phase 4：布局和导航（第8天）

#### 任务 4.1：侧边栏组件

创建 `frontend/src/components/layout/Sidebar.tsx`：

```tsx
import React from 'react'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  LineChart,
  PlayCircle,
  Radio,
  Shield,
  Settings,
  Sparkles,
  History,
  Database,
} from 'lucide-react'

interface NavItem {
  id: string
  label: string
  icon: React.ReactNode
  badge?: string
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: '仪表盘', icon: <LayoutDashboard className="h-5 w-5" /> },
  { id: 'backtest', label: '策略回测', icon: <LineChart className="h-5 w-5" /> },
  { id: 'sim-trading', label: '模拟交易', icon: <PlayCircle className="h-5 w-5" /> },
  { id: 'live-trading', label: '实盘交易', icon: <Radio className="h-5 w-5" />, badge: 'LIVE' },
  { id: 'risk-center', label: '风控中心', icon: <Shield className="h-5 w-5" /> },
  { id: 'optimizer', label: 'AI优化', icon: <Sparkles className="h-5 w-5" /> },
  { id: 'history', label: '回测历史', icon: <History className="h-5 w-5" /> },
  { id: 'data', label: '数据管理', icon: <Database className="h-5 w-5" /> },
  { id: 'settings', label: '系统设置', icon: <Settings className="h-5 w-5" /> },
]

interface SidebarProps {
  currentPage: string
  onNavigate: (page: string) => void
}

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onNavigate }) => {
  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-border bg-card">
      {/* Logo */}
      <div className="flex h-16 items-center border-b border-border px-6">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
            <LineChart className="h-5 w-5 text-white" />
          </div>
          <span className="text-lg font-bold">期货量化</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="space-y-1 p-4">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
              currentPage === item.id
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            )}
          >
            {item.icon}
            <span>{item.label}</span>
            {item.badge && (
              <span className={cn(
                "ml-auto rounded-full px-2 py-0.5 text-xs font-semibold",
                currentPage === item.id
                  ? "bg-white/20 text-white"
                  : "bg-danger text-white"
              )}>
                {item.badge}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 border-t border-border p-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center">
            <span className="text-sm font-medium">U</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">用户</p>
            <p className="text-xs text-muted-foreground">v2.0.0</p>
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
```

#### 任务 4.2：Header 组件

创建 `frontend/src/components/layout/Header.tsx`：

```tsx
import React from 'react'
import { Bell, Search, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

interface HeaderProps {
  title: string
  subtitle?: string
  onRefresh?: () => void
  loading?: boolean
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  onRefresh,
  loading = false,
}) => {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      {/* Title */}
      <div>
        <h1 className="text-xl font-semibold">{title}</h1>
        {subtitle && (
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索..."
            className="w-64 pl-9"
          />
        </div>

        {/* Refresh */}
        {onRefresh && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={loading}
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          </Button>
        )}

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-4 w-4" />
          <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-danger" />
        </Button>

        {/* Status */}
        <Badge variant="success">系统正常</Badge>
      </div>
    </header>
  )
}

// 需要导入 cn
import { cn } from '@/lib/utils'

export default Header
```

#### 任务 4.3：主布局组件

创建 `frontend/src/components/layout/MainLayout.tsx`：

```tsx
import React, { useState } from 'react'
import { Sidebar } from './Sidebar'
import { Header } from './Header'

interface MainLayoutProps {
  children: React.ReactNode
  title: string
  subtitle?: string
  onRefresh?: () => void
  loading?: boolean
}

export const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  title,
  subtitle,
  onRefresh,
  loading,
}) => {
  return (
    <div className="min-h-screen bg-background">
      <Sidebar currentPage={title.toLowerCase()} onNavigate={() => {}} />
      <div className="pl-64">
        <Header
          title={title}
          subtitle={subtitle}
          onRefresh={onRefresh}
          loading={loading}
        />
        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  )
}

export default MainLayout
```

---

### Phase 5：API 服务层（第9-10天）

#### 任务 5.1：API 客户端

创建 `frontend/src/services/api.ts`：

```typescript
import axios, { AxiosError, AxiosInstance } from 'axios'

const api: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加 token 等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default api
```

#### 任务 5.2：行情服务

创建 `frontend/src/services/market.ts`：

```typescript
import api from './api'

export interface SymbolInfo {
  symbol: string
  name: string
  exchange: string
  multiplier: number
  margin_rate: number
  start_date: string
  end_date: string
  count: number
}

export interface KlineData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export const marketService = {
  // 获取所有品种
  getSymbols: (): Promise<SymbolInfo[]> => {
    return api.get('/market/symbols')
  },

  // 获取K线数据
  getKline: (
    symbol: string,
    period: string = '日线',
    startDate?: string,
    endDate?: string,
    limit: number = 2000
  ): Promise<KlineData[]> => {
    return api.get('/market/kline', {
      params: { symbol, period, start_date: startDate, end_date: endDate, limit },
    })
  },

  // 获取实时行情（WebSocket）
  subscribeRealtime: (symbols: string[], onData: (data: any) => void) => {
    const ws = new WebSocket(`ws://localhost:8100/ws/market`)

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: 'subscribe', symbols }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      onData(data)
    }

    return () => ws.close()
  },
}
```

#### 任务 5.3：回测服务

创建 `frontend/src/services/backtest.ts`：

```typescript
import api from './api'

export interface BacktestConfig {
  strategy: string
  symbol: string
  start_date: string
  end_date: string
  initial_capital: number
  params: Record<string, any>
}

export interface BacktestResult {
  // 概览
  total_return: number
  annual_return: number
  max_drawdown: number
  sharpe_ratio: number
  win_rate: number
  profit_factor: number

  // 交易统计
  total_trades: number
  winning_trades: number
  losing_trades: number
  avg_profit: number
  avg_loss: number

  // 时序数据
  equity_curve: { date: string; equity: number; drawdown: number }[]
  trades: TradeRecord[]
}

export interface TradeRecord {
  id: number
  symbol: string
  direction: 'long' | 'short'
  entry_time: string
  entry_price: number
  exit_time: string
  exit_price: number
  quantity: number
  pnl: number
  pnl_percent: number
  exit_reason: string
}

export const backtestService = {
  // 获取策略列表
  getStrategies: (): Promise<string[]> => {
    return api.get('/backtest/strategies')
  },

  // 获取策略参数定义
  getStrategyParams: (strategy: string): Promise<any> => {
    return api.get(`/backtest/strategies/${strategy}/params`)
  },

  // 运行回测
  runBacktest: (config: BacktestConfig): Promise<BacktestResult> => {
    return api.post('/backtest/run', config)
  },

  // 获取回测历史
  getHistory: (): Promise<any[]> => {
    return api.get('/backtest/history')
  },
}
```

#### 任务 5.4：后端 API 路由

创建 `backend/routers/market.py`：

```python
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_manager import get_data_status, load_from_database, load_minute_from_database
from config import INSTRUMENTS

router = APIRouter()

@router.get("/symbols")
async def get_symbols():
    """获取所有可用交易品种"""
    df_status = get_data_status()
    result = []

    if not df_status.empty:
        for _, row in df_status.iterrows():
            symbol = row['symbol']
            inst = INSTRUMENTS.get(symbol, {})
            result.append({
                "symbol": symbol,
                "name": inst.get('name', symbol),
                "exchange": inst.get('exchange', ''),
                "multiplier": inst.get('multiplier', 1),
                "margin_rate": inst.get('margin_rate', 0.1),
                "start_date": str(row['start_date']),
                "end_date": str(row['end_date']),
                "count": int(row['record_count']),
            })

    return result

@router.get("/kline")
async def get_kline(
    symbol: str,
    period: str = Query("日线"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 2000
):
    """获取K线数据"""
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = "2010-01-01"

    try:
        if period == "日线":
            df = load_from_database(symbol, start_date, end_date)
        else:
            period_map = {"5分钟": "5", "15分钟": "15", "30分钟": "30", "60分钟": "60"}
            p = period_map.get(period, "5")
            df = load_minute_from_database(symbol, p, start_date, end_date)

        if df.empty:
            return []

        df = df.sort_values('time').iloc[-limit:]

        # 格式化
        if period == "日线":
            df['time'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
        else:
            df['time'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        return df[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

创建 `backend/routers/backtest.py`：

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()

class BacktestConfig(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 1000000
    params: Dict[str, Any] = {}

@router.get("/strategies")
async def get_strategies():
    """获取可用策略列表"""
    # 从 strategies 目录读取
    strategies_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'strategies'
    )

    strategies = []
    if os.path.exists(strategies_dir):
        for f in os.listdir(strategies_dir):
            if f.endswith('.py') and not f.startswith('_'):
                strategies.append(f[:-3])

    return strategies

@router.get("/strategies/{strategy}/params")
async def get_strategy_params(strategy: str):
    """获取策略参数定义"""
    # 动态导入策略模块获取参数定义
    # 这里需要根据你的策略结构来实现
    return {"params": []}

@router.post("/run")
async def run_backtest(config: BacktestConfig):
    """运行回测"""
    try:
        # 调用现有的回测引擎
        from engine import BacktestEngine
        from data_manager import load_from_database
        from config import INSTRUMENTS

        # 加载数据
        df = load_from_database(config.symbol, config.start_date, config.end_date)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available")

        # 获取品种信息
        inst = INSTRUMENTS.get(config.symbol, {})

        # 运行回测（这里需要根据你的引擎接口调整）
        # engine = BacktestEngine(...)
        # result = engine.run(df, config.params)

        # 返回模拟结果
        return {
            "total_return": 0.15,
            "annual_return": 0.25,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.5,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "total_trades": 100,
            "winning_trades": 55,
            "losing_trades": 45,
            "avg_profit": 5000,
            "avg_loss": -3000,
            "equity_curve": [],
            "trades": [],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history():
    """获取回测历史"""
    return []
```

---

### Phase 6：页面开发（第11-15天）

#### 任务 6.1：仪表盘页面

创建 `frontend/src/pages/Dashboard.tsx`：

```tsx
import React, { useEffect, useState } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { MetricCard } from '@/components/shared/MetricCard'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { CandlestickChart } from '@/components/charts/CandlestickChart'
import { marketService, SymbolInfo, KlineData } from '@/services/market'
import {
  TrendingUp,
  Activity,
  DollarSign,
  BarChart3,
  Loader2,
} from 'lucide-react'

const Dashboard: React.FC = () => {
  const [symbols, setSymbols] = useState<SymbolInfo[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState<string>('')
  const [klineData, setKlineData] = useState<KlineData[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    if (selectedSymbol) {
      loadKline(selectedSymbol)
    }
  }, [selectedSymbol])

  const loadData = async () => {
    try {
      const data = await marketService.getSymbols()
      setSymbols(data)
      if (data.length > 0) {
        setSelectedSymbol(data[0].symbol)
      }
    } catch (error) {
      console.error('Failed to load symbols:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadKline = async (symbol: string) => {
    try {
      const data = await marketService.getKline(symbol, '日线', undefined, undefined, 200)
      setKlineData(data)
    } catch (error) {
      console.error('Failed to load kline:', error)
    }
  }

  if (loading) {
    return (
      <MainLayout title="仪表盘">
        <div className="flex h-96 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </MainLayout>
    )
  }

  return (
    <MainLayout title="仪表盘" subtitle="系统概览与快速入口" onRefresh={loadData}>
      {/* 指标卡片 */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="账户净值"
          value={1250000}
          format="currency"
          change={0.08}
          changeLabel="本月收益"
          icon={<DollarSign className="h-5 w-5" />}
        />
        <MetricCard
          title="今日盈亏"
          value={12500}
          format="currency"
          valueClassName="text-profit"
          icon={<TrendingUp className="h-5 w-5" />}
        />
        <MetricCard
          title="持仓品种"
          value={3}
          format="none"
          subtitle="RB, IF, I"
          icon={<Activity className="h-5 w-5" />}
        />
        <MetricCard
          title="今日交易"
          value={8}
          format="none"
          subtitle="5盈 / 3亏"
          icon={<BarChart3 className="h-5 w-5" />}
        />
      </div>

      {/* 主图表区 */}
      <div className="mt-6 grid gap-6 lg:grid-cols-7">
        <Card className="lg:col-span-5">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>行情概览</span>
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="rounded-md border border-border bg-card px-3 py-1 text-sm"
              >
                {symbols.map((s) => (
                  <option key={s.symbol} value={s.symbol}>
                    {s.symbol} - {s.name}
                  </option>
                ))}
              </select>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <CandlestickChart data={klineData} height={400} />
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>品种列表</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-[400px] overflow-auto">
              {symbols.map((s) => (
                <div
                  key={s.symbol}
                  onClick={() => setSelectedSymbol(s.symbol)}
                  className={`flex cursor-pointer items-center justify-between rounded-lg border p-3 transition-colors ${
                    selectedSymbol === s.symbol
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:bg-secondary'
                  }`}
                >
                  <div>
                    <p className="font-medium">{s.symbol}</p>
                    <p className="text-sm text-muted-foreground">{s.name}</p>
                  </div>
                  <p className="text-sm text-muted-foreground">{s.count} 条</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  )
}

export default Dashboard
```

#### 任务 6.2：回测页面（核心页面）

创建 `frontend/src/pages/Backtest.tsx`：

```tsx
import React, { useState, useEffect } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table'
import { MetricCard } from '@/components/shared/MetricCard'
import { CandlestickChart, TradeMarker } from '@/components/charts/CandlestickChart'
import { EquityChart } from '@/components/charts/EquityChart'
import { marketService, SymbolInfo, KlineData } from '@/services/market'
import { backtestService, BacktestResult, TradeRecord } from '@/services/backtest'
import { formatNumber, formatPercent, formatCurrency } from '@/lib/utils'
import { Play, Download, Loader2 } from 'lucide-react'

const Backtest: React.FC = () => {
  // 状态
  const [symbols, setSymbols] = useState<SymbolInfo[]>([])
  const [strategies, setStrategies] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [klineData, setKlineData] = useState<KlineData[]>([])

  // 配置
  const [config, setConfig] = useState({
    strategy: '',
    symbol: '',
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    initialCapital: 1000000,
  })

  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    try {
      const [symbolsData, strategiesData] = await Promise.all([
        marketService.getSymbols(),
        backtestService.getStrategies(),
      ])
      setSymbols(symbolsData)
      setStrategies(strategiesData)

      if (symbolsData.length > 0) {
        setConfig(prev => ({ ...prev, symbol: symbolsData[0].symbol }))
      }
      if (strategiesData.length > 0) {
        setConfig(prev => ({ ...prev, strategy: strategiesData[0] }))
      }
    } catch (error) {
      console.error('Failed to load initial data:', error)
    }
  }

  const runBacktest = async () => {
    setLoading(true)
    try {
      // 加载K线数据
      const kline = await marketService.getKline(
        config.symbol,
        '日线',
        config.startDate,
        config.endDate
      )
      setKlineData(kline)

      // 运行回测
      const backtest = await backtestService.runBacktest({
        strategy: config.strategy,
        symbol: config.symbol,
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: config.initialCapital,
        params: {},
      })
      setResult(backtest)
    } catch (error) {
      console.error('Backtest failed:', error)
    } finally {
      setLoading(false)
    }
  }

  // 将交易记录转换为图表标记
  const getTradeMarkers = (): TradeMarker[] => {
    if (!result?.trades) return []

    const markers: TradeMarker[] = []
    result.trades.forEach((trade) => {
      // 入场标记
      markers.push({
        time: trade.entry_time,
        position: trade.direction === 'long' ? 'belowBar' : 'aboveBar',
        color: trade.direction === 'long' ? '#EF4444' : '#10B981',
        shape: trade.direction === 'long' ? 'arrowUp' : 'arrowDown',
        text: `${trade.direction === 'long' ? '买' : '卖'} ${trade.entry_price}`,
      })
      // 出场标记
      markers.push({
        time: trade.exit_time,
        position: trade.direction === 'long' ? 'aboveBar' : 'belowBar',
        color: trade.pnl >= 0 ? '#EF4444' : '#10B981',
        shape: 'circle',
        text: `平 ${trade.exit_price}`,
      })
    })
    return markers
  }

  return (
    <MainLayout title="策略回测" subtitle="历史数据回测与分析">
      <div className="space-y-6">
        {/* 配置区 */}
        <Card>
          <CardHeader>
            <CardTitle>回测配置</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-6">
              <div>
                <label className="text-sm font-medium">策略</label>
                <Select
                  value={config.strategy}
                  onValueChange={(v) => setConfig(prev => ({ ...prev, strategy: v }))}
                >
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="选择策略" />
                  </SelectTrigger>
                  <SelectContent>
                    {strategies.map((s) => (
                      <SelectItem key={s} value={s}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium">品种</label>
                <Select
                  value={config.symbol}
                  onValueChange={(v) => setConfig(prev => ({ ...prev, symbol: v }))}
                >
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="选择品种" />
                  </SelectTrigger>
                  <SelectContent>
                    {symbols.map((s) => (
                      <SelectItem key={s.symbol} value={s.symbol}>
                        {s.symbol} - {s.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium">开始日期</label>
                <Input
                  type="date"
                  value={config.startDate}
                  onChange={(e) => setConfig(prev => ({ ...prev, startDate: e.target.value }))}
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-sm font-medium">结束日期</label>
                <Input
                  type="date"
                  value={config.endDate}
                  onChange={(e) => setConfig(prev => ({ ...prev, endDate: e.target.value }))}
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-sm font-medium">初始资金</label>
                <Input
                  type="number"
                  value={config.initialCapital}
                  onChange={(e) => setConfig(prev => ({ ...prev, initialCapital: Number(e.target.value) }))}
                  className="mt-1"
                />
              </div>

              <div className="flex items-end">
                <Button onClick={runBacktest} disabled={loading} className="w-full">
                  {loading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  运行回测
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 结果区 */}
        {result && (
          <>
            {/* 概览指标 */}
            <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
              <MetricCard
                title="总收益率"
                value={result.total_return}
                format="percent"
                valueClassName={result.total_return >= 0 ? 'text-profit' : 'text-loss'}
              />
              <MetricCard
                title="年化收益"
                value={result.annual_return}
                format="percent"
                valueClassName={result.annual_return >= 0 ? 'text-profit' : 'text-loss'}
              />
              <MetricCard
                title="最大回撤"
                value={result.max_drawdown}
                format="percent"
                valueClassName="text-loss"
              />
              <MetricCard
                title="夏普比率"
                value={result.sharpe_ratio}
                decimals={2}
              />
              <MetricCard
                title="胜率"
                value={result.win_rate}
                format="percent"
              />
              <MetricCard
                title="盈亏比"
                value={result.profit_factor}
                decimals={2}
              />
            </div>

            {/* 详细结果 Tabs */}
            <Card>
              <Tabs defaultValue="chart">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <TabsList>
                      <TabsTrigger value="chart">K线交易图</TabsTrigger>
                      <TabsTrigger value="equity">资金曲线</TabsTrigger>
                      <TabsTrigger value="trades">交易记录</TabsTrigger>
                      <TabsTrigger value="stats">统计分析</TabsTrigger>
                    </TabsList>
                    <Button variant="outline" size="sm">
                      <Download className="mr-2 h-4 w-4" />
                      导出报告
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <TabsContent value="chart">
                    <CandlestickChart
                      data={klineData}
                      markers={getTradeMarkers()}
                      height={500}
                      showVolume={true}
                      showMA={true}
                    />
                  </TabsContent>

                  <TabsContent value="equity">
                    <EquityChart
                      data={result.equity_curve}
                      height={400}
                      showDrawdown={true}
                      initialCapital={config.initialCapital}
                    />
                  </TabsContent>

                  <TabsContent value="trades">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>序号</TableHead>
                          <TableHead>方向</TableHead>
                          <TableHead>入场时间</TableHead>
                          <TableHead>入场价</TableHead>
                          <TableHead>出场时间</TableHead>
                          <TableHead>出场价</TableHead>
                          <TableHead>数量</TableHead>
                          <TableHead>盈亏</TableHead>
                          <TableHead>收益率</TableHead>
                          <TableHead>出场原因</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {result.trades.map((trade, idx) => (
                          <TableRow key={trade.id}>
                            <TableCell>{idx + 1}</TableCell>
                            <TableCell>
                              <Badge variant={trade.direction === 'long' ? 'profit' : 'loss'}>
                                {trade.direction === 'long' ? '多' : '空'}
                              </Badge>
                            </TableCell>
                            <TableCell className="font-mono text-sm">{trade.entry_time}</TableCell>
                            <TableCell className="font-mono">{formatNumber(trade.entry_price)}</TableCell>
                            <TableCell className="font-mono text-sm">{trade.exit_time}</TableCell>
                            <TableCell className="font-mono">{formatNumber(trade.exit_price)}</TableCell>
                            <TableCell>{trade.quantity}</TableCell>
                            <TableCell className={`font-mono font-medium ${trade.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                              {formatCurrency(trade.pnl)}
                            </TableCell>
                            <TableCell className={`font-mono ${trade.pnl_percent >= 0 ? 'text-profit' : 'text-loss'}`}>
                              {formatPercent(trade.pnl_percent)}
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline">{trade.exit_reason}</Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TabsContent>

                  <TabsContent value="stats">
                    <div className="grid gap-4 md:grid-cols-3">
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-base">收益统计</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">总交易次数</span>
                            <span className="font-mono">{result.total_trades}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">盈利次数</span>
                            <span className="font-mono text-profit">{result.winning_trades}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">亏损次数</span>
                            <span className="font-mono text-loss">{result.losing_trades}</span>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="text-base">盈亏统计</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">平均盈利</span>
                            <span className="font-mono text-profit">{formatCurrency(result.avg_profit)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">平均亏损</span>
                            <span className="font-mono text-loss">{formatCurrency(result.avg_loss)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">盈亏比</span>
                            <span className="font-mono">{result.profit_factor.toFixed(2)}</span>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="text-base">风险统计</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">最大回撤</span>
                            <span className="font-mono text-loss">{formatPercent(result.max_drawdown)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">夏普比率</span>
                            <span className="font-mono">{result.sharpe_ratio.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">收益回撤比</span>
                            <span className="font-mono">
                              {(result.annual_return / Math.abs(result.max_drawdown)).toFixed(2)}
                            </span>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </>
        )}
      </div>
    </MainLayout>
  )
}

export default Backtest
```

#### 任务 6.3：App 入口

更新 `frontend/src/App.tsx`：

```tsx
import React, { useState } from 'react'
import Dashboard from '@/pages/Dashboard'
import Backtest from '@/pages/Backtest'
// import LiveTrading from '@/pages/LiveTrading'
// import SimTrading from '@/pages/SimTrading'
// import RiskCenter from '@/pages/RiskCenter'
// import Optimizer from '@/pages/Optimizer'
// import Settings from '@/pages/Settings'
import { Sidebar } from '@/components/layout/Sidebar'

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />
      case 'backtest':
        return <Backtest />
      // case 'live-trading':
      //   return <LiveTrading />
      // case 'sim-trading':
      //   return <SimTrading />
      // case 'risk-center':
      //   return <RiskCenter />
      // case 'optimizer':
      //   return <Optimizer />
      // case 'settings':
      //   return <Settings />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="pl-64">
        {renderPage()}
      </div>
    </div>
  )
}

export default App
```

---

### Phase 7：联调测试（第16-18天）

#### 任务 7.1：后端主入口更新

更新 `backend/main.py`，取消注释并导入路由：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="期货量化系统 API",
    description="支持回测、交易、数据管理的后端 API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "期货量化系统 API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 导入路由
from backend.routers import market, backtest
app.include_router(market.router, prefix="/api/market", tags=["行情数据"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["回测"])
```

#### 任务 7.2：创建启动脚本

创建 `start_dev.bat`：

```batch
@echo off
echo Starting Backend...
start cmd /k "cd /d D:\期货\回测改造 && python -m uvicorn backend.main:app --reload --port 8100"

echo Starting Frontend...
timeout /t 3
start cmd /k "cd /d D:\期货\回测改造\frontend && npm run dev"

echo.
echo ==========================================
echo Backend: http://localhost:8100
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8100/docs
echo ==========================================
```

#### 任务 7.3：测试清单

创建 `TEST_CHECKLIST.md`：

```markdown
# 测试清单

## 基础功能

- [ ] 前端能正常启动 (npm run dev)
- [ ] 后端能正常启动 (uvicorn)
- [ ] API 文档可访问 (/docs)
- [ ] 前后端通信正常 (CORS)

## 仪表盘页面

- [ ] 品种列表加载正常
- [ ] K线图显示正常
- [ ] 指标卡片数据正确
- [ ] 切换品种图表更新

## 回测页面

- [ ] 策略列表加载正常
- [ ] 品种选择正常
- [ ] 日期选择正常
- [ ] 回测运行成功
- [ ] 结果概览显示正确
- [ ] K线交易图标记正确
- [ ] 资金曲线显示正确
- [ ] 交易记录表格正常
- [ ] 导出功能正常

## 样式检查

- [ ] 米白背景色正确 (#FAFAF8)
- [ ] 紫色强调色正确 (#7C3AED)
- [ ] 字体显示正常 (Inter, JetBrains Mono)
- [ ] 响应式布局正常
- [ ] 深浅色对比度合适
```

---

## 四、后续页面（可选扩展）

以下页面可在基础功能完成后逐步添加：

1. **模拟交易页面** (`SimTrading.tsx`)
2. **实盘交易页面** (`LiveTrading.tsx`)
3. **风控中心页面** (`RiskCenter.tsx`)
4. **AI参数优化页面** (`Optimizer.tsx`)
5. **系统设置页面** (`Settings.tsx`)
6. **回测历史页面** (`History.tsx`)
7. **数据管理页面** (`DataManagement.tsx`)

每个页面的实现模式与 Dashboard 和 Backtest 类似。

---

## 五、注意事项

### 5.1 执行顺序

1. 严格按 Phase 顺序执行
2. 每个 Phase 完成后进行测试
3. 遇到问题及时记录并修复

### 5.2 常见问题

| 问题 | 解决方案 |
|------|----------|
| 模块导入错误 | 检查 Python 路径配置 |
| CORS 错误 | 确认后端 CORS 配置正确 |
| 图表不显示 | 检查数据格式是否正确 |
| 样式不生效 | 确认 Tailwind 配置正确 |

### 5.3 开发命令

```bash
# 启动后端
cd "D:\期货\回测改造"
python -m uvicorn backend.main:app --reload --port 8100

# 启动前端
cd "D:\期货\回测改造\frontend"
npm run dev

# 构建生产版本
npm run build
```

---

**文档版本**：v1.0
**创建日期**：2026-01-11
**预计完成**：2026-01-31
