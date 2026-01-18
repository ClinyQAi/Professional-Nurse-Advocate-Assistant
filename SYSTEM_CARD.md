---
title: "Building the PNA Assistant: A Nurse's Guide to AI Resilience Tools"
authors:
  - user: NurseCitizenDeveloper
    guest: true
tags:
  - nlp
  - healthcare
  - gemma
  - rag
  - ethical-ai
  - system-card
---

# Building the PNA Assistant: A Nurse's Guide to AI Resilience Tools

**"AI isn't here to replace nurses. It's here to be built by us."**

The resilience of the nursing workforce is a global priority. In the UK, the **Professional Nurse Advocate (PNA)** role has been a game-changer, using the **A-EQUIP model** to provide restorative clinical supervision. But with staffing pressures, accessing a trained PNA when you need one most can be a challenge.

What if we could build a tool to bridge that gap? Not to replace the human supervisor, but to provide a safe, 24/7 space for reflection?

Enter the **PNA Assistant**—an open-source AI research tool built by a nurse, for nurses.

## 🏥 What is the PNA Assistant?

The [PNA Assistant](https://huggingface.co/spaces/NurseCitizenDeveloper/PNA-Assistant) is a specialized AI agent designed to simulate the "Restorative Clinical Supervision" function of a Professional Nurse Advocate. 

Unlike generic chatbots, it doesn't just "chat." It is grounded in the **A-EQUIP Model** (Advocating for Educating for Quality Improvement and Practice) and strictly adheres to the official NHS PNA Handbook.

### Key Capabilities
*   **Restorative Dialogue**: Asking reflective questions to help nurses process emotional challenges.
*   **Policy Grounding**: Retrieving exact guidance from the A-EQUIP framework (Normative, Formative, and Restorative).
*   **Safety First**: Hard-coded safety rails to distinguish educational support from clinical/crisis intervention.

---

## ⚙️ Under the Hood: The "Nurse Citizen Developer" Stack

We built this tool using a **Retrieval-Augmented Generation (RAG)** architecture, hosted entirely on Hugging Face Spaces using ZeroGPU.

### 1. The Brain: Gemma 2 2B IT 🧠
We chose Google's **Gemma 2 2B IT** (Instruction Tuned) as our foundation. Why?
*   **Efficient**: It runs fast on renewable cloud resources.
*   **Capable**: It has excellent reasoning capabilities for its size.
*   **Open**: It aligns with our philosophy of open healthcare research.

### 2. The Knowledge: RAG 📚
To prevent hallucinations (AI making things up), we don't rely solely on the model's training data. Instead, we use RAG:
1.  **Knowledge Base**: We embedded the *PNA Handbook* and *A-EQUIP Guidance*.
2.  **Retrieval**: When you ask a question, the system searches these documents for the answer.
3.  **Generation**: The AI uses that official text to construct its response.

### 3. The Persona: "Compassionate Instruction Tuning" ❤️
The "secret sauce" is the System Prompt. We didn't just tell it to be helpful; we told it to be a **Nurse**.
*   *Legacy Challenge*: AI can sound robotic or overly solution-focused.
*   *Our Solution*: We instructed the model to prioritize **Validation** and **Reflection** over immediate problem-solving, mirroring the core skills of a PNA.

---

## 🚀 Why "Citizen Development" Matters

This project proves that you don't need a Computer Science degree to build impactful health AI. You need clinical expertise and curiosity.

By taking ownership of the technology, we ensured that **Nursing Values**—compassion, safety, and advocacy—were baked into the code from Day 1. This isn't just an AI tool; it's a statement that nurses are ready to lead in the digital era.

## 🎓 Research & Availability

The PNA Assistant is available today as a **Research Artifact**. We invite nursing researchers, educators, and digital health innovators to test it, break it, and build upon it.

*   **Try the Model**: [Hugging Face Space](https://huggingface.co/spaces/NurseCitizenDeveloper/PNA-Assistant)
*   **Read the Code**: [GitHub Repository](https://github.com/ClinyQAi/Professional-Nurse-Advocate-Assistant)

*Disclaimer: This tool is for educational and research simulation purposes only. It does not provide clinical diagnosis or crisis mental health support.*
