// Bundle the existing HTSW checker and harness into this repo's ignored build/.
// Neither source checkout is modified. Requires their installed dependencies.
// HTSW_ROOT=/path/to/htsw HTSW_SIM_ROOT=/path/to/htsw-simulator node sim/build-tools.mjs
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const htsw = process.env.HTSW_ROOT ?? "/home/opc/htsw";
const harness = process.env.HTSW_SIM_ROOT ?? "/home/opc/htsw-simulator";
const require = createRequire(path.join(htsw, "package.json"));
const { build } = require("esbuild");
const plugins = [{ name: "htsw-source", setup(b) {
    // jsonc-parser's UMD build uses relative dynamic require calls.
    b.onResolve({ filter: /^jsonc-parser$/ }, () => ({
        path: require.resolve("jsonc-parser"), external: true,
    }));
    b.onResolve({ filter: /^htsw(?:\/.*)?$/ }, ({ path: name }) => ({
        path: path.join(htsw, "language/src", name === "htsw" ? "" : name.slice(5), "index.ts"),
    }));
} }];
for (const [entry, name] of [
    [path.join(htsw, "cli/src/main.ts"), "htsw"],
    [path.join(harness, "src/index.ts"), "simulator"],
]) {
    await build({ entryPoints: [entry], outfile: path.join(root, `build/tools/${name}.mjs`),
        bundle: true, platform: "node", format: "esm", plugins,
        banner: { js: "import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);" },
    });
}
console.log("Built local HTSW checker and simulator; source checkouts unchanged.");
