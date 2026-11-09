import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit


class Inspector(HTMLParser):
	def __init__(self, base_url):
		super().__init__(convert_charrefs=True)
		self.base_url = base_url
		self.title_depth = 0
		self.title_text = []
		self.iframes = 0
		self.hidden_iframes = 0
		self.images = 0
		self.submit_buttons = 0
		self.hidden_inputs = 0
		self.password_inputs = 0
		self.meta_refreshes = 0
		self.forms = []
		self.form_stack = []
		self.in_script = False
		self.inline_scripts = []

	def handle_starttag(self, tag, attrs):
		a = dict(attrs)
		if tag == 'title':
			self.title_depth += 1
		elif tag == 'img':
			self.images += 1
		elif tag == 'iframe':
			self.iframes += 1
			style = (a.get('style') or '').replace(' ', '').lower()
			if ('hidden' in a or a.get('aria-hidden', '').lower() == 'true'
					or 'display:none' in style or 'visibility:hidden' in style
					or a.get('width') == '0' or a.get('height') == '0'):
				self.hidden_iframes += 1
		elif tag == 'form':
			action = a.get('action', '') or ''
			target = urljoin(self.base_url, action)
			base_host = urlsplit(self.base_url).hostname
			dest_host = urlsplit(target).hostname
			cross_domain = bool(
				base_host and dest_host and base_host.lower() != dest_host.lower()
			)
			form = {
				'action': action,
				'destination': target,
				'cross_domain': cross_domain,
				'password': False,
			}
			self.forms.append(form)
			self.form_stack.append(form)
		elif tag == 'input':
			typ = (a.get('type') or 'text').lower()
			if typ == 'password':
				self.password_inputs += 1
				if self.form_stack:
					self.form_stack[-1]['password'] = True
			if typ == 'hidden':
				self.hidden_inputs += 1
			if typ in ('submit', 'image'):
				self.submit_buttons += 1
		elif tag == 'button':
			if (a.get('type') or 'submit').lower() == 'submit':
				self.submit_buttons += 1
		elif tag == 'meta':
			if (a.get('http-equiv') or '').lower() == 'refresh':
				self.meta_refreshes += 1
		elif tag == 'script':
			self.in_script = not bool(a.get('src'))

	def handle_endtag(self, tag):
		if tag == 'title' and self.title_depth:
			self.title_depth -= 1
		if tag == 'form' and self.form_stack:
			self.form_stack.pop()
		if tag == 'script':
			self.in_script = False

	def handle_data(self, data):
		if self.title_depth:
			self.title_text.append(data)
		if self.in_script:
			self.inline_scripts.append(data)


def inspect(path, base_url):
	parser = Inspector(base_url)
	parser.feed(path.read_text(encoding='utf-8-sig', errors='replace'))
	title = ''.join(parser.title_text).strip()
	return {
		'file': str(path),
		'model_six_features': {
			'HasTitle': int(bool(title)),
			'NoOfiFrame': parser.iframes,
			'HasSubmitButton': int(parser.submit_buttons > 0),
			'HasHiddenFields': int(parser.hidden_inputs > 0),
			'HasPasswordField': int(parser.password_inputs > 0),
			'NoOfImage': parser.images,
		},
		'additional_review_features': {
			'CrossDomainForms': sum(f['cross_domain'] for f in parser.forms),
			'CrossDomainPasswordForms': sum(
				f['cross_domain'] and f['password'] for f in parser.forms
			),
			'HiddenIframes': parser.hidden_iframes,
			'MetaRefreshes': parser.meta_refreshes,
		},
		'forms': parser.forms,
		'notes': [
			'Static HTML inspection only; dynamically created elements are not examined.',
			'Cross-domain forms, iframes, and redirects are not proof of phishing.',
			'Feature names resemble the UCI dataset but extraction equivalence has not been verified. Do not feed these new features into the old trained model.',
		],
	}


def main():
	parser = argparse.ArgumentParser(
		description='Validate features on local HTML files (no network access).'
	)
	parser.add_argument('files', nargs='+', type=Path)
	parser.add_argument('--base-url', default='https://demo.example')
	parser.add_argument('--out', default='results/v4_3_validation.json')
	args = parser.parse_args()

	results = []
	for path in args.files:
		if not path.is_file():
			parser.error(f'找不到檔案：{path}')
		if path.stat().st_size > 2 * 1024 * 1024:
			parser.error(f'檔案過大：{path}')
		results.append(inspect(path, args.base_url))

	output_path = Path(args.out)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
	)
	print(json.dumps(results, ensure_ascii=False, indent=2))
	print(f'結果已儲存：{output_path}')


if __name__ == '__main__':
	main()
