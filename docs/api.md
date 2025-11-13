# 🧠 Neurosense REST API Documentation

**Base URL:** `http://localhost:5000/api`

---

## 1️⃣ Health Check
**Endpoint:** `GET /health`  
**Description:** Verify that the backend is running.  

### ✅ Example Response (200)
```json
{
  "status": "UP",
  "message": "Neurosense API is running."
}
