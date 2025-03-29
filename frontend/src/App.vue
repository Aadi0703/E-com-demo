<template>
  <div class="container">
    <h2>🛍️ Demo E-Commerce Chat Assistant</h2>

    <!-- Chat Messages -->
    <div class="chat-window">
      <div
        v-for="(chat, index) in chatHistory"
        :key="index"
        :class="['chat-message', chat.role === 'User' ? 'user' : 'bot']"
      >
        <span class="label">{{ chat.role === 'User' ? '🧑 You' : '🤖 Assistant' }}</span>
        <div class="message" v-html="formatMessage(chat.content)"></div>
      </div>
    </div>

    <!-- Input Controls -->
    <div class="controls">
      <input
        v-model="query"
        placeholder="Type your question..."
        @keyup.enter="fetchResponse"
      />
      <button @click="fetchResponse">Ask</button>
      <button @click="clearChat" class="clear-btn">Clear</button>
    </div>

    <div v-if="loading" class="loading">⏳ Thinking...</div>
  </div>
</template>

<script>
const API_URL = "http://127.0.0.1:8000/query";

export default {
  data() {
    return {
      query: "",
      chatHistory: [],
      loading: false
    };
  },
  methods: {
    async fetchResponse() {
      if (!this.query.trim()) return;
      this.loading = true;

      try {
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: this.query,
            history: this.chatHistory
          })
        });

        const data = await res.json();

        this.chatHistory.push({ role: "User", content: this.query });
        this.chatHistory.push({ role: "Bot", content: data.response });
        this.query = "";
      } catch (error) {
        console.error("Error fetching response:", error);
      }

      this.loading = false;
    },

    clearChat() {
      this.chatHistory = [];
    },

    formatMessage(content) {
      return content
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\n/g, "<br/>");
    }
  }
};
</script>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  font-family: "Segoe UI", sans-serif;
}

h2 {
  text-align: center;
  color: #444;
  margin-bottom: 1rem;
}

.chat-window {
  max-height: 500px;
  overflow-y: auto;
  border: 1px solid #e0e0e0;
  background-color: #f9f9f9;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.chat-message {
  margin-bottom: 1rem;
  padding: 0.8rem;
  border-radius: 6px;
  transition: all 0.3s;
}

.chat-message:hover {
  background-color: #f0f0f0;
}

.chat-message.user {
  background-color: #e1f5fe;
  border-left: 4px solid #0288d1;
}

.chat-message.bot {
  background-color: #e8f5e9;
  border-left: 4px solid #43a047;
}

.label {
  font-weight: bold;
  color: #333;
  margin-bottom: 0.4rem;
  display: block;
}

.message {
  white-space: pre-wrap;
  color: #333;
}

.controls {
  display: flex;
  gap: 10px;
  margin-top: 1rem;
}

input {
  flex: 1;
  padding: 0.7rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  font-size: 1rem;
}

button {
  padding: 0.7rem 1.2rem;
  font-size: 1rem;
  border: none;
  border-radius: 6px;
  background-color: #0288d1;
  color: white;
  cursor: pointer;
}

button:hover {
  background-color: #0277bd;
}

.clear-btn {
  background-color: #757575;
}

.clear-btn:hover {
  background-color: #616161;
}

.loading {
  text-align: center;
  font-size: 1.1rem;
  color: #555;
  margin-top: 0.8rem;
}
</style>
