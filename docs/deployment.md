# 部署上线指南

本指南帮助你将 RAG 系统部署到公网，让面试官能够直接体验你的项目。

---

## 方案对比

| 方案 | 成本 | 难度 | 推荐指数 |
|-----|------|-----|---------|
| Render（免费） | 0元 | ⭐ | ⭐⭐⭐⭐⭐ |
| Railway（免费额度） | 0元 | ⭐ | ⭐⭐⭐⭐ |
| 阿里云学生机 | 9元/月 | ⭐⭐ | ⭐⭐⭐ |
| 腾讯云学生机 | 10元/月 | ⭐⭐ | ⭐⭐⭐ |

---

## 方案一：Render 部署（推荐，完全免费）

### Step 1: 准备 GitHub 仓库

```bash
# 在项目目录下
git init
git add .
git commit -m "Initial commit"

# 创建 GitHub 仓库后
git remote add origin https://github.com/你的用户名/rag-system.git
git branch -M main
git push -u origin main
```

### Step 2: 注册 Render

1. 访问 https://render.com
2. 点击 "Get Started" 注册（可以用 GitHub 登录）

### Step 3: 创建服务

1. 点击 "New +" → "Web Service"
2. 连接你的 GitHub 仓库
3. 填写配置：

```
Name: rag-system
Region: Oregon (US West) 或 Singapore
Branch: main
Runtime: Docker
Plan: Free
```

### Step 4: 设置环境变量

在 "Environment Variables" 中添加：

```
QIANFAN_AK = 你的百度千帆AccessKey
QIANFAN_SK = 你的百度千帆SecretKey
```

### Step 5: 部署

点击 "Create Web Service"，等待几分钟即可完成。

部署成功后，你会获得一个地址：
```
https://rag-system-xxx.onrender.com
```

访问 `https://rag-system-xxx.onrender.com/docs` 即可看到 API 文档。

---

## 方案二：Railway 部署

### Step 1: 安装 Railway CLI

```bash
npm install -g @railway/cli
```

### Step 2: 登录并部署

```bash
railway login
railway init
railway up
```

### Step 3: 设置环境变量

```bash
railway variables set QIANFAN_AK=你的key
railway variables set QIANFAN_SK=你的secret
```

---

## 方案三：云服务器部署

### Step 1: 购买服务器

推荐阿里云学生机（约9元/月）：
- 配置：2核4G
- 系统：Ubuntu 20.04

### Step 2: 连接服务器

```bash
ssh root@你的服务器IP
```

### Step 3: 安装 Docker

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 启动 Docker
systemctl start docker
systemctl enable docker
```

### Step 4: 部署项目

```bash
# 创建目录
mkdir -p /opt/rag
cd /opt/rag

# 上传代码（在本地执行）
scp -r ./AI_TEST1/* root@服务器IP:/opt/rag/

# 在服务器上构建并运行
docker build -t rag-api .
docker run -d -p 8000:8000 \
  -e QIANFAN_AK=你的key \
  -e QIANFAN_SK=你的secret \
  --name rag-api \
  rag-api
```

### Step 5: 访问服务

```
http://你的服务器IP:8000/docs
```

---

## 本地测试 Docker

在部署前，建议先本地测试：

```bash
# 构建镜像
docker build -t rag-api .

# 本地运行
docker run -p 8000:8000 \
  -e QIANFAN_AK=你的key \
  -e QIANFAN_SK=你的secret \
  rag-api

# 测试访问
# http://localhost:8000/docs
```

---

## 常见问题

### Q: 部署后访问不了？

检查：
1. 服务是否启动成功
2. 端口是否正确（8000）
3. 云服务器安全组是否开放端口

### Q: Render 免费版有什么限制？

- 750小时/月
- 服务15分钟无请求会休眠
- 休眠后首次请求需要等待启动（约30秒）

### Q: 如何绑定自定义域名？

1. 在 Render 控制台添加自定义域名
2. 在域名服务商处添加 CNAME 记录指向 Render 地址

---

## 部署成功后

### 1. 更新简历

```
项目链接：https://rag-system-xxx.onrender.com/docs
API文档：https://rag-system-xxx.onrender.com/docs
```

### 2. 测试功能

```bash
# 健康检查
curl https://你的地址/health

# 测试查询
curl -X POST https://你的地址/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能？"}'
```

### 3. 面试时展示

直接打开网页演示，比口头描述有说服力10倍。
