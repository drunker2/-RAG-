#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 系统 Web 界面
使用 Streamlit 构建，提供用户友好的问答界面

运行方式：
    streamlit run app.py
"""

import streamlit as st
import requests
import time
from typing import Dict, Any

# 页面配置
st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_api_url() -> str:
    """获取 API 地址"""
    # 如果设置了环境变量则使用，否则使用本地
    import os
    return os.getenv("API_URL", "http://localhost:8000")


def query_api(question: str) -> Dict[str, Any]:
    """调用 API 进行查询"""
    api_url = get_api_url()

    try:
        response = requests.post(
            f"{api_url}/query",
            json={"question": question, "show_sources": True, "max_sources": 3},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到服务器，请确保 API 服务已启动"}
    except requests.exceptions.Timeout:
        return {"error": "请求超时，请稍后重试"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}


def check_health() -> bool:
    """检查 API 健康状态"""
    api_url = get_api_url()
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# 主页面
def main():
    # 标题
    st.markdown('<h1 class="main-header">🤖 RAG 智能问答系统</h1>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")

        # API 状态
        st.subheader("系统状态")
        if check_health():
            st.success("✅ 服务运行正常")
        else:
            st.error("❅ 服务未连接")
            st.info("请确保运行: `python api.py`")

        st.divider()

        # 使用说明
        st.header("📖 使用说明")
        st.markdown("""
        1. 在输入框中输入问题
        2. 点击"提问"按钮
        3. 系统会从知识库检索相关文档
        4. AI 基于检索结果生成回答

        **提示**：问题越具体，回答越准确
        """)

        st.divider()

        # 示例问题
        st.header("💡 示例问题")
        examples = [
            "什么是人工智能？",
            "RAG系统是如何工作的？",
            "向量检索的原理是什么？",
            "BM25算法有什么优势？"
        ]

        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.question = example

    # 主内容区
    col1, col2 = st.columns([2, 1])

    with col1:
        # 问题输入
        question = st.text_area(
            "📝 请输入您的问题：",
            height=100,
            placeholder="例如：什么是RAG系统？它有哪些优势？",
            value=st.session_state.get("question", "")
        )

        # 提交按钮
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            submit = st.button("🚀 提问", type="primary", use_container_width=True)
        with col_btn2:
            clear = st.button("🗑️ 清空", use_container_width=True)

        if clear:
            st.session_state.question = ""
            st.rerun()

        # 处理查询
        if submit and question.strip():
            with st.spinner("🤔 正在思考中..."):
                start_time = time.time()
                result = query_api(question)
                elapsed = time.time() - start_time

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                # 显示回答
                st.markdown("### 💬 回答")

                # 置信度指示
                confidence = result.get("confidence", 0)
                if confidence >= 0.7:
                    confidence_class = "confidence-high"
                    confidence_text = "高"
                elif confidence >= 0.4:
                    confidence_class = "confidence-medium"
                    confidence_text = "中"
                else:
                    confidence_class = "confidence-low"
                    confidence_text = "低"

                st.markdown(f"""
                <div class="source-box">
                    <p>{result.get('answer', '未获取到回答')}</p>
                </div>
                <p>置信度: <span class="{confidence_class}">{confidence_text} ({confidence:.0%})</span> |
                耗时: {elapsed:.2f}秒</p>
                """, unsafe_allow_html=True)

                # 显示来源
                sources = result.get("sources", [])
                if sources:
                    st.markdown("### 📚 参考来源")
                    for i, source in enumerate(sources[:3], 1):
                        content = source.get("content", "")
                        if len(content) > 200:
                            content = content[:200] + "..."

                        with st.expander(f"来源 {i}", expanded=(i == 1)):
                            st.markdown(f"""
                            <div class="source-box">
                                <small>{content}</small>
                            </div>
                            """, unsafe_allow_html=True)

                # 如果问题被优化过
                if result.get("query_was_optimized"):
                    st.info(f"🔍 问题已优化: {result.get('optimized_question', question)}")

        elif submit:
            st.warning("⚠️ 请输入问题")

    with col2:
        # 统计信息
        st.header("📊 系统信息")

        # 模拟数据（实际应从 API 获取）
        st.metric("文档数量", "128", delta="+12")
        st.metric("今日查询", "47", delta="+5")
        st.metric("平均响应", "1.2s", delta="-0.3s")

        st.divider()

        # 技术栈
        st.header("🔧 技术栈")
        tech_stack = """
        - **LLM**: 百度千帆 ERNIE-3.5
        - **向量库**: ChromaDB
        - **嵌入模型**: all-MiniLM-L6-v2
        - **检索方式**: 混合检索 (BM25+向量)
        - **后端**: FastAPI
        - **前端**: Streamlit
        """
        st.markdown(tech_stack)


# 运行
if __name__ == "__main__":
    main()
