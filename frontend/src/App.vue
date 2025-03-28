<template>
  <div>
    <h2>Demo E-commerce </h2>

    <div class="chat-window">
      <div
        v-for="(chat, index) in chatHistory"
        :key="index"
        class="chat-message"
        v-html="formatMessage(chat.content)"
      ></div>
    </div>

    <div class="controls">
      <input v-model="query" placeholder="Ask a question..." />
      <button @click="fetchResponse">Ask</button>
    </div>

    <div v-if="loading">Loading...</div>
  </div>
</template>

<script>
const API_URL = "http://127.0.0.1:8000/query"; // Adjust if hosted elsewhere

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
      if (!this.query) return;
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

    formatMessage(content) {
      return content
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\n/g, "<br/>");
    }
  }
};
</script>

<style>
.chat-window {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}
.chat-message {
  margin-bottom: 8px;
}
.controls {
  display: flex;
  gap: 8px;
}
</style>
