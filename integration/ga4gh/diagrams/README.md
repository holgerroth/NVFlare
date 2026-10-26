# Integration diagrams

The main README and code walkthrough embed PNG images so they display in Markdown viewers without Mermaid
support. SVG copies are available for zooming and export. Edit the corresponding `.mmd` source and regenerate
both formats when changing a diagram.

| Diagram | Editable source | PNG | SVG |
|---|---|---|---|
| Overview | [overview.mmd](overview.mmd) | [overview.png](overview.png) | [overview.svg](overview.svg) |
| Code walkthrough | [integration-points.mmd](integration-points.mmd) | [integration-points.png](integration-points.png) | [integration-points.svg](integration-points.svg) |

The `adapter-*` assets preserve the separate optional inbound WES architecture used in `inbound-wes.md` and `adapter-integration-points.md`. The main diagrams show the recipe calling an existing site WES service.

Blue identifies GA4GH WES/TES components and connections; green identifies NVFLARE APIs and federation.
Gray identifies the research platform, data, and ordinary analysis code. Dashed boxes identify operational
ownership: coordinator/platform operator versus each data owner's approved environment. Cloud resources can
belong inside either boundary; the boxes are not physical host or network boundaries. The overview's data stores
illustrate the intended deployment, while the synthetic pilot uses arrays.

Avoid trailing semicolons on Mermaid style
lines: a numeric hex color immediately followed by a semicolon can be interpreted as an entity by the parser.

Regenerate from the integration directory using Node.js and Mermaid CLI 12.0.0:

```bash
for diagram in overview integration-points; do
  for format in svg png; do
    npx --yes --package @mermaid-js/mermaid-cli@12.0.0 mmdc \
      -i "diagrams/$diagram.mmd" -o "diagrams/$diagram.$format" \
      -c diagrams/mermaid-config.json -b white --size 2000 --no-font-embed
  done
done
```

Mermaid CLI uses headless Chrome. If using an existing browser installation, provide its executable path in
a local Puppeteer JSON configuration and pass `-p /path/to/puppeteer-config.json`. The browser path is specific
to the machine and is not part of these source files. Rendering is a documentation maintenance step; readers
and the WES/TES services need no Node.js or Mermaid dependencies.
