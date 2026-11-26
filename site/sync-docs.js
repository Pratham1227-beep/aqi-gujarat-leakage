import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const docsSrc = path.resolve(__dirname, '../docs');
const contentDest = path.resolve(__dirname, 'src/content/docs');

if (!fs.existsSync(contentDest)) {
  fs.mkdirSync(contentDest, { recursive: true });
}

const titles = {
  'protocol.md': 'Experimental Protocol',
  'data_availability.md': 'Data Availability Audit',
  'cpcb_breakpoints.md': 'CPCB Breakpoints Reference',
  'decisions.md': 'Architectural Decisions',
  'paper_outline.md': 'Manuscript Outline'
};

const orders = {
  'protocol.md': 1,
  'data_availability.md': 2,
  'cpcb_breakpoints.md': 3,
  'decisions.md': 4,
  'paper_outline.md': 5
};

const files = fs.readdirSync(docsSrc).filter(f => f.endsWith('.md'));

for (const file of files) {
  let content = fs.readFileSync(path.join(docsSrc, file), 'utf-8');
  const title = titles[file] || file.replace('.md', '');
  const order = orders[file] || 99;
  
  // Ensure frontmatter
  if (!content.startsWith('---')) {
    content = `---\ntitle: "${title}"\norder: ${order}\n---\n\n` + content;
  }
  
  fs.writeFileSync(path.join(contentDest, file), content, 'utf-8');
  console.log(`Synced ${file} to src/content/docs/`);
}
