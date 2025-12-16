# Mikhail Builds Website Template for Beginners

A modern, beginner-friendly Next.js template designed for rapid web development with Cursor IDE integration.

## 🚀 Tech Stack

- **Framework**: Next.js 16 with App Router
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS v4 with custom animations
- **Development**: Turbopack for fast builds
- **Linting**: ESLint with Next.js configuration
- **Build Tool**: PostCSS with Autoprefixer

## ✨ Features

- **Modern UI Components**: Pre-built Hero component with animated gradient backgrounds
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **TypeScript Support**: Full type checking and IntelliSense
- **Custom Animations**: Blob animations and smooth transitions
- **Developer Experience**: Optimized with Turbopack for lightning-fast development
- **Cursor IDE Integration**: Designed to work seamlessly with Cursor's AI agent

## 🛠 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or higher)
- [Cursor IDE](https://cursor.sh/) (recommended)

### Installation

1. **Download Cursor IDE**
   - Visit [cursor.sh](https://cursor.sh/) and download the latest version

2. **Clone the repository**
   ```bash
   git clone https://github.com/mwijanarko1/template.git
   cd template
   ```

3. **Install dependencies**
   ```bash
   npm install
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open Cursor Composer**
   - Press `CMD + I` to open the Composer
   - Make sure to select the "Agent" mode

6. **Use the AI Agent**
   - Copy the contents of the `PROMPT.txt` file and paste it into the Composer Agent
   - Let the AI generate your project structure and components
   - Chat with the agent to build features - it will handle the coding for you

7. **Fix any issues**
   - If errors occur, copy/paste the error or screenshot it
   - Ask the agent to fix the specific error

## 📁 Project Structure

```
template/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout component
│   │   ├── page.tsx           # Home page
│   │   ├── globals.css        # Global styles
│   │   └── favicon.ico        # App favicon
│   └── components/
│       └── Hero.tsx           # Hero section component
├── tailwind.config.js         # Tailwind CSS configuration
├── next.config.mjs           # Next.js configuration
├── postcss.config.mjs        # PostCSS configuration
├── eslint.config.mjs         # ESLint configuration
└── package.json              # Dependencies and scripts
```

## 🎨 Customization

### Styling
- Modify `tailwind.config.js` to add custom colors, fonts, or animations
- Update `src/app/globals.css` for global styles
- Components use Tailwind utility classes for easy customization

### Components
- Add new components in `src/components/`
- Import and use them in your pages
- Follow the existing Hero component pattern

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Start Production Server
```bash
npm start
```

The template is ready to deploy to Vercel, Netlify, or any other hosting platform that supports Next.js.

## 🤝 Contributing

This template is designed to be extended and customized. Feel free to:
- Add new components
- Modify the styling
- Extend functionality
- Share your improvements

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
