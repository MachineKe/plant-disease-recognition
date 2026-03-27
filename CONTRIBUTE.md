# Contributing to Plant Disease Recognition

Welcome to the Plant Disease Recognition project! This repository currently features a Python/Flask backend for image-based disease detection, with plans to add a Next.js frontend in the future. We appreciate your interest in contributing and aim to make the process clear and accessible.

---

## 1. Introduction

- **Backend:** Python/Flask app for plant disease recognition.
- **Frontend (Planned):** Next.js web interface for user interaction and visualization.

---

## 2. General Contribution Guidelines

- **Branching:** Always create a new branch for your work. Never push directly to `main`.
- **Pull Requests:** All changes must be submitted via PR and require review/approval before merging.
- **No Direct Pushes:** Direct pushes to `main` are strictly prohibited.

---

## 3. Setting Up the Development Environment

### Python Prerequisites
- Python 3.8+ is required.
- Install [virtualenv](https://virtualenv.pypa.io/en/latest/) for isolated environments.

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/machineke/plant-disease-recognition.git
   cd plant-disease-recognition
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```bash
   python web_app/app.py
   ```

---

## 4. Code Style and Linting

- Follow [PEP8](https://pep8.org/) guidelines.
- Use [flake8](https://flake8.pycqa.org/) for linting:
  ```bash
  flake8 web_app/
  ```
- Use [black](https://black.readthedocs.io/) for formatting:
  ```bash
  black web_app/
  ```
- Write clear docstrings for all functions and classes.

---

## 5. Testing

- All tests are located in `web_app/test_model.py` or similar test files.
- Use [pytest](https://docs.pytest.org/) for running tests:
  ```bash
  pytest web_app/
  ```
- Add tests for new features and bug fixes.

---

## 6. Git Workflow

- **Branch Naming:** Use descriptive names, e.g., `feature/model-improvement`, `bugfix/image-upload`.
- **Commit Messages:** Write concise, meaningful messages. Reference ticket IDs if applicable.
- **Stay Up to Date:** Regularly pull from `main` to avoid conflicts:
  ```bash
  git fetch origin
  git pull origin main
  ```

---

## 7. Pull Request Process

- Open a PR from your branch to `main`.
- Ensure your PR includes:
  - Clear title and description
  - Reference to any relevant tickets
  - Checklist:
    - [ ] Code follows style guidelines
    - [ ] Tests added/updated
    - [ ] All tests pass
    - [ ] Documentation updated (if needed)
- Request review from at least one other contributor.
- Address feedback and update your PR as needed.

---

## 8. Next.js Frontend Contributions

### 8.1 Prerequisites
- **Node.js**: Install [Node.js](https://nodejs.org/) (version 16+ recommended).
- **npm or Yarn**: Use [npm](https://www.npmjs.com/) (comes with Node.js) or [Yarn](https://yarnpkg.com/) for package management.
- **Next.js Basics**: Familiarity with [Next.js](https://nextjs.org/docs) and React is helpful.

### 8.2 Setting Up the Next.js Development Environment
1. Navigate to the frontend directory (e.g., `nextjs_frontend/`).
2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```
3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```
4. Access the app at [http://localhost:3000](http://localhost:3000).

### 8.3 Code Style and Linting
- **JavaScript/TypeScript**: Use [ESLint](https://eslint.org/) and [Prettier](https://prettier.io/) for code quality and formatting.
- Run lint checks:
  ```bash
  npm run lint
  # or
  yarn lint
  ```
- Follow the project's `.eslintrc` and `.prettierrc` configurations.
- Use TypeScript for new files unless otherwise specified.

### 8.4 Testing Guidelines
- Use [Jest](https://jestjs.io/) and [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) for unit and integration tests.
- Add tests for new components, pages, and features.
- Run tests locally:
  ```bash
  npm run test
  # or
  yarn test
  ```
- Ensure all tests pass before submitting a PR.

### 8.5 Directory Structure & Where to Contribute
- **`pages/`**: Add new routes/pages here.
- **`components/`**: Add reusable UI components here.
- **`styles/`**: Add CSS/SCSS files here.
- **`public/`**: Static assets (images, icons, etc.).
- **`utils/`**: Helper functions and utilities.
- Follow existing patterns and naming conventions.

### 8.6 Submitting a Pull Request (PR) for the Next.js Frontend
- **Branching**: Create a feature branch from `main`, e.g., `feature/add-login-page`.
- **Checklist Before PR**:
  - [ ] Code follows ESLint/Prettier guidelines
  - [ ] Tests added/updated
  - [ ] All tests pass
  - [ ] Documentation updated (if needed)
  - [ ] No breaking changes to backend integration
- **PR Process**:
  1. Open a PR to `main`.
  2. Provide a clear title and description.
  3. Reference any relevant issues/tickets.
  4. Request review from at least one contributor.
  5. Address feedback and update your PR as needed.

### 8.7 Integration Notes: Next.js Frontend & Flask Backend
- The Next.js frontend communicates with the Flask backend via HTTP API endpoints.
- Update API URLs in the frontend as needed (e.g., `/api/plant-disease`).
- Ensure CORS is enabled on the Flask backend for frontend requests.
- Coordinate changes to API contracts with backend maintainers.
- Test end-to-end functionality after frontend/backend changes.

---

---

## 9. Additional Resources

- **Documentation:** See `README.md` and `plant_disease_recognition_report.md` for project details.
- **Ticketing:** Track issues and features via GitHub Issues or Trello (if applicable).
- **Contacts:** For questions, reach out via GitHub Discussions or contact the maintainers listed in `README.md`.

---

Thank you for contributing to Plant Disease Recognition! Your efforts help us build a robust and user-friendly platform for plant health analysis.
