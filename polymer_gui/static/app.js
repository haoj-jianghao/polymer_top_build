const pegRepeatSdf = `PEG_repeat_monomer
  polymer_gui

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4500    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.9000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$`;

const sketch = {
  atoms: [],
  bonds: [],
  element: "C",
  tool: "atom",
  formalCharge: 0,
  selectedAtom: null,
  selectedBond: null,
  pointerStart: null,
  previewPoint: null,
  isDragging: false,
};

const canvas = document.getElementById("sketcher");
const ctx = canvas.getContext("2d");

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function setActiveButton(containerId, attr, value) {
  document.querySelectorAll(`#${containerId} button`).forEach(button => {
    button.classList.toggle("active", button.dataset[attr] === value);
  });
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function atomAt(point) {
  for (let i = sketch.atoms.length - 1; i >= 0; i -= 1) {
    const atom = sketch.atoms[i];
    if (Math.hypot(atom.x - point.x, atom.y - point.y) <= 20) return atom;
  }
  return null;
}

function distanceToSegment(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) return Math.hypot(point.x - a.x, point.y - a.y);
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSq));
  return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy));
}

function bondAt(point) {
  for (let i = sketch.bonds.length - 1; i >= 0; i -= 1) {
    const bond = sketch.bonds[i];
    const a = sketch.atoms.find(atom => atom.id === bond.a);
    const b = sketch.atoms.find(atom => atom.id === bond.b);
    if (a && b && distanceToSegment(point, a, b) <= 8) return bond;
  }
  return null;
}

function addAtom(element, x, y) {
  sketch.atoms.push({ id: sketch.atoms.length + 1, element, x, y, formalCharge: 0 });
}

function addBond(a, b) {
  if (!a || !b || a.id === b.id) return;
  const existing = sketch.bonds.find(bond =>
    (bond.a === a.id && bond.b === b.id) || (bond.a === b.id && bond.b === a.id)
  );
  if (existing) {
    existing.order = existing.order === 1 ? 2 : 1;
  } else {
    sketch.bonds.push({ a: a.id, b: b.id, order: 1 });
  }
}

function renumberAtoms() {
  const oldToNew = new Map();
  sketch.atoms.forEach((atom, index) => {
    oldToNew.set(atom.id, index + 1);
    atom.id = index + 1;
  });
  sketch.bonds = sketch.bonds
    .map(bond => ({ a: oldToNew.get(bond.a), b: oldToNew.get(bond.b), order: bond.order }))
    .filter(bond => bond.a && bond.b);
}

function eraseAtom(atom) {
  sketch.atoms = sketch.atoms.filter(item => item.id !== atom.id);
  sketch.bonds = sketch.bonds.filter(bond => bond.a !== atom.id && bond.b !== atom.id);
  sketch.selectedAtom = null;
  sketch.selectedBond = null;
  renumberAtoms();
}

function eraseBond(bond) {
  sketch.bonds = sketch.bonds.filter(item => item !== bond);
  sketch.selectedBond = null;
}

function deleteSelected() {
  if (sketch.selectedAtom) {
    eraseAtom(sketch.selectedAtom);
    setStatus("Deleted selected atom and its bonds.");
  } else if (sketch.selectedBond) {
    eraseBond(sketch.selectedBond);
    setStatus("Deleted selected bond.");
  } else {
    setStatus("Select an atom or bond first, then delete it.");
  }
  drawSketch();
}

function elementColor(element) {
  return { C: "#111111", O: "#cf2f2f", N: "#2356c4", S: "#9a6b00", H: "#64717b", Cl: "#11824c" }[element] || "#111111";
}

function formalChargeLabel(charge) {
  if (!charge) return "";
  const magnitude = Math.abs(charge);
  return (magnitude === 1 ? "" : String(magnitude)) + (charge > 0 ? "+" : "−");
}


function updateStructureLine() {
  const line = document.getElementById("structureLine");
  const formalCharge = sketch.atoms.reduce((sum, atom) => sum + (atom.formalCharge || 0), 0);
  if (line) {
    line.value = "Drawn_monomer    atoms=" + sketch.atoms.length
      + "    bonds=" + sketch.bonds.length + "    charge=" + formalCharge;
  }
}

function molBlockFromSketch() {
  if (sketch.atoms.length === 0) throw new Error("Draw at least one atom first.");
  const lines = ["Drawn_monomer", "  polymer_gui", ""];
  lines.push(`${String(sketch.atoms.length).padStart(3)}${String(sketch.bonds.length).padStart(3)}  0  0  0  0            999 V2000`);
  for (const atom of sketch.atoms) {
    const x = ((atom.x - 80) / 55).toFixed(4).padStart(10);
    const y = ((canvas.height - atom.y - 80) / 55).toFixed(4).padStart(10);
    lines.push(`${x}${y}${"0.0000".padStart(10)} ${atom.element.padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0`);
  }
  for (const bond of sketch.bonds) {
    lines.push(`${String(bond.a).padStart(3)}${String(bond.b).padStart(3)}${String(bond.order).padStart(3)}  0  0  0  0`);
  }
  const chargedAtoms = sketch.atoms.filter(atom => atom.formalCharge);
  for (let start = 0; start < chargedAtoms.length; start += 8) {
    const chunk = chargedAtoms.slice(start, start + 8);
    const pairs = chunk
      .map(atom => String(atom.id).padStart(4) + String(atom.formalCharge).padStart(4))
      .join("");
    lines.push("M  CHG" + String(chunk.length).padStart(3) + pairs);
  }
  lines.push("M  END", "$$$$");
  return lines.join("\n");
}

function updateMolTextFromSketch() {
  const sdfBox = document.getElementById("sdf");
  if (!sdfBox || document.activeElement === sdfBox) return;
  sdfBox.value = sketch.atoms.length > 0 ? molBlockFromSketch() : "";
}

function drawBondLine(a, b, order) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  const ox = (-dy / length) * 4;
  const oy = (dx / length) * 4;
  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 2.4;
  if (order === 2) {
    ctx.beginPath();
    ctx.moveTo(a.x + ox, a.y + oy);
    ctx.lineTo(b.x + ox, b.y + oy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(a.x - ox, a.y - oy);
    ctx.lineTo(b.x - ox, b.y - oy);
    ctx.stroke();
  } else {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawSketch() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fdfefe";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#eef3f7";
  ctx.lineWidth = 1;
  for (let x = 32; x < canvas.width; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 32; y < canvas.height; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  for (const bond of sketch.bonds) {
    const a = sketch.atoms.find(atom => atom.id === bond.a);
    const b = sketch.atoms.find(atom => atom.id === bond.b);
    if (!a || !b) continue;
    if (sketch.selectedBond === bond) {
      ctx.strokeStyle = "rgba(44, 133, 205, 0.35)";
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    drawBondLine(a, b, bond.order);
  }

  if (sketch.isDragging && sketch.pointerStart && sketch.previewPoint) {
    ctx.strokeStyle = "#1d6dab";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(sketch.pointerStart.x, sketch.pointerStart.y);
    ctx.lineTo(sketch.previewPoint.x, sketch.previewPoint.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  for (const atom of sketch.atoms) {
    const selected = sketch.selectedAtom && sketch.selectedAtom.id === atom.id;
    if (selected) {
      ctx.fillStyle = "rgba(44, 133, 205, 0.18)";
      ctx.strokeStyle = "#1d6dab";
      ctx.lineWidth = 1.5;
      ctx.fillRect(atom.x - 22, atom.y - 19, 44, 38);
      ctx.strokeRect(atom.x - 22, atom.y - 19, 44, 38);
    }
    ctx.fillStyle = "#fdfefe";
    ctx.fillRect(atom.x - 14, atom.y - 12, 28, 20);
    ctx.fillStyle = elementColor(atom.element);
    ctx.font = "bold 17px Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(atom.element, atom.x, atom.y - 1);
    if (atom.formalCharge) {
      ctx.fillStyle = "#8b1e3f";
      ctx.font = "bold 12px Arial, sans-serif";
      ctx.fillText(formalChargeLabel(atom.formalCharge), atom.x + 15, atom.y - 12);
    }
    ctx.fillStyle = "#0b5cab";
    ctx.font = "12px Arial, sans-serif";
    ctx.fillText(String(atom.id), atom.x + 18, atom.y + 18);
  }
  updateStructureLine();
  updateMolTextFromSketch();
}

function loadMolIntoSketch(sdfText) {
  const lines = sdfText.split(/\r?\n/);
  const atomCount = Number.parseInt((lines[3] || "").slice(0, 3), 10);
  const bondCount = Number.parseInt((lines[3] || "").slice(3, 6), 10);
  if (!Number.isFinite(atomCount) || !Number.isFinite(bondCount)) return;
  sketch.atoms = [];
  sketch.bonds = [];
  for (let i = 0; i < atomCount; i += 1) {
    const line = lines[4 + i];
    const x = Number.parseFloat(line.slice(0, 10));
    const y = Number.parseFloat(line.slice(10, 20));
    const chargeCode = Number.parseInt(line.slice(36, 39).trim(), 10) || 0;
    const chargeFromCode = { 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3 };
    sketch.atoms.push({
      id: i + 1,
      element: line.slice(31, 34).trim(),
      x: 100 + x * 60,
      y: canvas.height / 2 - y * 60,
      formalCharge: chargeFromCode[chargeCode] || 0,
    });
  }
  for (let i = 0; i < bondCount; i += 1) {
    const line = lines[4 + atomCount + i];
    sketch.bonds.push({
      a: Number.parseInt(line.slice(0, 3), 10),
      b: Number.parseInt(line.slice(3, 6), 10),
      order: Number.parseInt(line.slice(6, 9), 10) || 1,
    });
  }
  for (let i = 4 + atomCount + bondCount; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line.startsWith("M  CHG")) continue;
    const fields = line.trim().split(/\s+/);
    const count = Number.parseInt(fields[2], 10) || 0;
    for (let pair = 0; pair < count; pair += 1) {
      const atomIndex = Number.parseInt(fields[3 + pair * 2], 10);
      const formalCharge = Number.parseInt(fields[4 + pair * 2], 10);
      const atom = sketch.atoms[atomIndex - 1];
      if (atom && Number.isFinite(formalCharge)) atom.formalCharge = formalCharge;
    }
  }
  sketch.selectedAtom = null;
  sketch.selectedBond = null;
  drawSketch();
}

function syncDrawingToTextarea() {
  document.getElementById("sdf").value = molBlockFromSketch();
  updateStructureLine();
}

function chooseConnectionAtom(index) {
  const previous = document.getElementById("previous_atom");
  const next = document.getElementById("next_atom");
  if (!previous.value || previous.value === "0") {
    previous.value = String(index);
  } else if (!next.value || next.value === "0" || next.value === previous.value) {
    next.value = String(index);
  } else {
    next.value = String(index);
  }
}

canvas.addEventListener("mousedown", event => {
  const point = canvasPoint(event);
  sketch.pointerStart = { x: point.x, y: point.y, atom: atomAt(point) };
  sketch.previewPoint = point;
  sketch.isDragging = false;
});

canvas.addEventListener("mousemove", event => {
  if (!sketch.pointerStart) return;
  const point = canvasPoint(event);
  sketch.previewPoint = point;
  sketch.isDragging = Math.hypot(point.x - sketch.pointerStart.x, point.y - sketch.pointerStart.y) > 8;
  drawSketch();
});

canvas.addEventListener("mouseup", event => {
  if (!sketch.pointerStart) return;
  const point = canvasPoint(event);
  const endAtom = atomAt(point);
  const startAtom = sketch.pointerStart.atom;
  const dragged = Math.hypot(point.x - sketch.pointerStart.x, point.y - sketch.pointerStart.y) > 8;

  if (sketch.tool === "charge") {
    if (endAtom) {
      endAtom.formalCharge = sketch.formalCharge;
      sketch.selectedAtom = endAtom;
      sketch.selectedBond = null;
      setStatus(
        "Assigned formal charge " + formalChargeLabel(sketch.formalCharge)
        + (sketch.formalCharge ? "" : "0") + " to atom " + endAtom.id + "."
      );
    } else {
      setStatus("Choose an existing atom to assign its formal charge.");
    }
    sketch.pointerStart = null;
    sketch.previewPoint = null;
    sketch.isDragging = false;
    drawSketch();
    return;
  }

  if (dragged) {
    let a = startAtom;
    let b = endAtom;
    if (!a) {
      addAtom("C", sketch.pointerStart.x, sketch.pointerStart.y);
      a = sketch.atoms[sketch.atoms.length - 1];
    }
    if (!b) {
      addAtom("C", point.x, point.y);
      b = sketch.atoms[sketch.atoms.length - 1];
    }
    addBond(a, b);
    sketch.selectedAtom = b;
    sketch.selectedBond = null;
  } else if (endAtom) {
    endAtom.element = sketch.element;
    sketch.selectedAtom = endAtom;
    sketch.selectedBond = null;
    chooseConnectionAtom(endAtom.id);
  } else {
    const hitBond = bondAt(point);
    if (hitBond) {
      hitBond.order = hitBond.order === 1 ? 2 : 1;
      sketch.selectedBond = hitBond;
      sketch.selectedAtom = null;
    } else {
      addAtom(sketch.element, point.x, point.y);
      sketch.selectedAtom = sketch.atoms[sketch.atoms.length - 1];
      sketch.selectedBond = null;
    }
  }

  sketch.pointerStart = null;
  sketch.previewPoint = null;
  sketch.isDragging = false;
  drawSketch();
});

canvas.addEventListener("contextmenu", event => {
  event.preventDefault();
  const point = canvasPoint(event);
  const hitAtom = atomAt(point);
  if (hitAtom) {
    eraseAtom(hitAtom);
    setStatus("Deleted atom and its bonds.");
  } else {
    const hitBond = bondAt(point);
    if (hitBond) {
      eraseBond(hitBond);
      setStatus("Deleted bond.");
    }
  }
  sketch.pointerStart = null;
  sketch.previewPoint = null;
  sketch.isDragging = false;
  drawSketch();
});

window.addEventListener("mouseup", () => {
  if (!sketch.pointerStart) return;
  sketch.pointerStart = null;
  sketch.previewPoint = null;
  sketch.isDragging = false;
  drawSketch();
});

document.querySelectorAll("#atomPalette button").forEach(button => {
  button.addEventListener("click", () => {
    sketch.tool = "atom";
    sketch.element = button.dataset.element;
    setActiveButton("atomPalette", "element", sketch.element);
    document.querySelectorAll("#chargePalette button").forEach(item => item.classList.remove("active"));
  });
});

document.querySelectorAll("#chargePalette button").forEach(button => {
  button.addEventListener("click", () => {
    sketch.tool = "charge";
    sketch.formalCharge = Number.parseInt(button.dataset.charge, 10) || 0;
    setActiveButton("chargePalette", "charge", String(sketch.formalCharge));
    if (sketch.selectedAtom) {
      sketch.selectedAtom.formalCharge = sketch.formalCharge;
      drawSketch();
    }
    setStatus("Formal-charge tool active. Click an atom to assign " + formalChargeLabel(sketch.formalCharge) + (sketch.formalCharge ? "" : "0") + ".");
  });
});

document.getElementById("clearSketch").addEventListener("click", () => {
  sketch.atoms = [];
  sketch.bonds = [];
  sketch.selectedAtom = null;
  sketch.selectedBond = null;
  document.getElementById("sdf").value = "";
  drawSketch();
  setStatus("Cleared the drawing.");
});

document.getElementById("deleteSelected").addEventListener("click", deleteSelected);
window.addEventListener("keydown", event => {
  if (event.key !== "Delete" && event.key !== "Backspace") return;
  const tag = document.activeElement ? document.activeElement.tagName : "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  event.preventDefault();
  deleteSelected();
});

function formData() {
  const data = new FormData();
  for (const id of [
    "sdf",
    "workflow",
    "previous_atom",
    "next_atom",
    "reference_dp",
    "dp",
    "job_name",
  ]) {
    data.append(id, document.getElementById(id).value);
  }
  data.append("run_external", document.getElementById("run_external").checked ? "true" : "false");
  return data;
}

async function postForm(url) {
  const response = await fetch(url, { method: "POST", body: formData() });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function renderPreview(payload) {
  document.getElementById("summary").textContent =
    payload.name + "\nFormula: " + payload.formula
    + "\nMolecular weight: " + payload.molecular_weight
    + "\nFormal charge: " + payload.formal_charge
    + "\nAtoms: " + payload.atoms.length
    + "\nBonds: " + payload.bonds.length;
  const atoms = document.getElementById("atoms");
  atoms.innerHTML = "";
  for (const atom of payload.atoms) {
    const div = document.createElement("button");
    div.type = "button";
    div.className = "atom";
    const charge = formalChargeLabel(atom.formal_charge);
    div.innerHTML = "<strong>" + atom.index + "</strong>" + atom.element
      + (charge ? '<span class="atom-charge">' + charge + "</span>" : "");
    div.addEventListener("click", () => chooseConnectionAtom(atom.index));
    atoms.appendChild(div);
  }
}

function renderBuild(payload) {
  const files = payload.files.map(file => {
    const name = file.split("/").pop();
    return `<a href="/${file}" download>${name}</a>`;
  }).join("\n");
  const blocked = payload.missing_tools.length
    ? `\nAmberTools blocked: missing ${payload.missing_tools.join(", ")}`
    : "\nAmberTools found.";
  document.getElementById("output").innerHTML =
    `Formula: ${payload.formula}<br>` +
    `Atoms: ${payload.atoms}<br>` +
    `Bonds: ${payload.bonds}<br>` +
    `Reference repeat units: ${JSON.stringify(payload.repeat_units)}<br>` +
    `Final charge: ${payload.final_charge}<br>` +
    `${blocked}<br><br>` +
    files.replaceAll("\n", "<br>");
}

document.getElementById("loadPeg").addEventListener("click", () => {
  document.getElementById("sdf").value = pegRepeatSdf;
  document.getElementById("previous_atom").value = "1";
  document.getElementById("next_atom").value = "3";
  loadMolIntoSketch(pegRepeatSdf);
  setStatus("Loaded PEG repeat monomer. The hidden reference oligomer size controls the AmberTools training structure.");
});

document.getElementById("pullMol").addEventListener("click", () => {
  try {
    syncDrawingToTextarea();
    setStatus("Copied the monomer drawing into MOL/SDF text.");
  } catch (err) {
    setStatus(err.message);
  }
});

document.getElementById("preview").addEventListener("click", async () => {
  try {
    if (!document.getElementById("sdf").value.trim() && sketch.atoms.length > 0) syncDrawingToTextarea();
    setStatus("Reading monomer...");
    const payload = await postForm("/api/preview");
    renderPreview(payload);
    setStatus("Preview ready. Click atom tiles to fill previous/next connection atoms.");
  } catch (err) {
    setStatus(err.message);
  }
});

document.getElementById("build").addEventListener("click", async () => {
  try {
    if (!document.getElementById("sdf").value.trim() && sketch.atoms.length > 0) syncDrawingToTextarea();
    setStatus("Generating hidden oligomer and polymer files...");
    const payload = await postForm("/api/build");
    renderBuild(payload);
    setStatus("Build finished.");
  } catch (err) {
    setStatus(err.message);
  }
});

drawSketch();
