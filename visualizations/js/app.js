document.addEventListener("DOMContentLoaded", () => {
  const filters = document.querySelectorAll(".arch-filter");
  const searchInput = document.getElementById("search");
  const modal = document.getElementById("modal");
  const viewer = document.getElementById("viewer");
  const modalTitle = document.getElementById("modalTitle");
  const connRoute = document.getElementById("connRoute");
  const connWeight = document.getElementById("connWeight");
  const connMeta = document.getElementById("connMeta");
  const edgeList = document.getElementById("edgeList");
  const showWeights = document.getElementById("showWeights");
  const showBiases = document.getElementById("showBiases");

  let activeType = "all";
  let zoom = 1;

  function templateId(src) {
    return "tpl-" + String(src).replace(/\//g, "-").replace(/\./g, "-");
  }

  function flattenArchitecture(type) {
    document.querySelectorAll("section").forEach((section) => {
      const flatGrid = section.querySelector(".architecture-grid");
      if (!flatGrid) return;

      flatGrid.innerHTML = "";
      section.querySelectorAll(".entry").forEach((entry) => {
        const title = entry.querySelector("h3")?.textContent || "";
        const card = entry.querySelector(`.card[data-type="${type}"]`);
        if (!card) return;

        const wrapper = document.createElement("div");
        wrapper.className = "card";
        wrapper.dataset.type = card.dataset.type || type;
        wrapper.dataset.search = card.dataset.search || "";
        if (card.dataset.src) wrapper.dataset.src = card.dataset.src;
        if (card.dataset.title) wrapper.dataset.title = card.dataset.title;
        wrapper.setAttribute("role", "button");
        wrapper.setAttribute("tabindex", "0");
        wrapper.innerHTML = `
          <div class="entry-label">${title}</div>
          ${card.innerHTML}
        `;
        flatGrid.appendChild(wrapper);
      });
    });
  }

  function filterCards() {
    const searchTerm = searchInput.value.toLowerCase().trim();

    document.querySelectorAll(".card").forEach((card) => {
      const type = card.dataset.type || "";
      const searchText = (card.dataset.search || "").toLowerCase();

      const typeMatch = activeType === "all" || type === activeType;
      const searchMatch = searchTerm === "" || searchText.includes(searchTerm);

      card.style.display = typeMatch && searchMatch ? "" : "none";
    });

    if (activeType !== "all") {
      document.querySelectorAll(".architecture-grid").forEach((grid) => {
        const visibleCards = [...grid.querySelectorAll(".card")].filter(
          (card) => card.style.display !== "none"
        );
        grid.style.display = visibleCards.length > 0 ? "grid" : "none";
      });
      return;
    }

    document.querySelectorAll(".entry").forEach((entry) => {
      const visibleCards = [...entry.querySelectorAll(".card")].filter(
        (card) => card.style.display !== "none"
      );
      entry.style.display = visibleCards.length > 0 ? "" : "none";
    });

    document.querySelectorAll("section").forEach((section) => {
      if (section.classList.contains("architecture-overview")) {
        section.style.display = "";
        return;
      }

      const visibleEntries = [...section.querySelectorAll(".entry")].filter(
        (entry) => entry.style.display !== "none"
      );
      section.style.display = visibleEntries.length > 0 ? "" : "none";
    });
  }

  function clearHighlight(svg) {
    svg.querySelectorAll(".edge").forEach((edge) => edge.classList.remove("dim"));
    svg.querySelectorAll(".neuron").forEach((neuron) => neuron.classList.remove("dim"));
    svg.querySelectorAll(".wlabel").forEach((label) => label.classList.remove("active"));
  }

  function highlightEdge(svg, edge) {
    clearHighlight(svg);
    svg.querySelectorAll(".edge").forEach((otherEdge) => {
      if (otherEdge !== edge) otherEdge.classList.add("dim");
    });

    const from = edge.dataset.from;
    const to = edge.dataset.to;
    svg.querySelectorAll(".neuron").forEach((neuron) => {
      const id = neuron.dataset.neuron;
      if (id !== from && id !== to) neuron.classList.add("dim");
    });

    const edgeId = edge.dataset.edge;
    if (edgeId !== undefined) {
      svg.querySelectorAll(".wlabel").forEach((label) => {
        label.classList.toggle("active", label.dataset.edge === edgeId);
      });
    }
  }

  function fmtWeight(value) {
    const numeric = Number(value);
    const sign = numeric >= 0 ? "+" : "-";
    return sign + Math.abs(numeric).toFixed(2);
  }

  function setZoom(value) {
    zoom = value;
    const svg = viewer.querySelector("svg");
    if (svg) svg.style.transform = "scale(" + zoom + ")";
  }

  function showConn(edge) {
    const numeric = Number(edge.dataset.weight);
    connRoute.textContent = edge.dataset.from + " → " + edge.dataset.to;
    connWeight.textContent = fmtWeight(numeric);
    connWeight.className = "wval " + (numeric >= 0 ? "pos" : "neg");
    connMeta.textContent =
      "Layer " + edge.dataset.layer + " · |w| = " + Math.abs(numeric).toFixed(4);
  }

  function hideConn() {
    connRoute.textContent = "Hover a connection";
    connWeight.textContent = "—";
    connWeight.className = "wval";
    connMeta.textContent =
      "Edge thickness scales with |weight|. Blue = positive, red = negative.";
  }

  function bindSvg(svg) {
    svg.classList.toggle("hide-weights", !showWeights.checked);
    svg.classList.toggle("hide-biases", !showBiases.checked);

    const edges = [...svg.querySelectorAll(".edge")];
    edgeList.innerHTML = "";

    edges.forEach((edge) => {
      const item = document.createElement("li");
      const numeric = Number(edge.dataset.weight);
      const cls = numeric >= 0 ? "pos" : "neg";
      item.innerHTML =
        `<span>${edge.dataset.from} → ${edge.dataset.to}</span>` +
        `<span class="w ${cls}">${fmtWeight(numeric)}</span>`;

      item.addEventListener("mouseenter", () => {
        highlightEdge(svg, edge);
        showConn(edge);
      });
      item.addEventListener("mouseleave", () => {
        clearHighlight(svg);
        hideConn();
      });

      edgeList.appendChild(item);

      edge.addEventListener("mouseenter", () => {
        highlightEdge(svg, edge);
        showConn(edge);
      });
      edge.addEventListener("mouseleave", () => {
        clearHighlight(svg);
        hideConn();
      });
    });

    if (!edges.length) {
      edgeList.innerHTML = '<li class="hint">Schematic network — no flat weights stored.</li>';
    }
  }

  function mountDiagram(src) {
    const template = document.getElementById(templateId(src));
    if (!template) return null;

    viewer.innerHTML = "";
    viewer.appendChild(template.content.cloneNode(true));
    return viewer.querySelector("svg");
  }

  async function openModal(src, title) {
    modalTitle.textContent = title;
    viewer.innerHTML = "Loading…";
    modal.classList.add("open");
    setZoom(1);
    hideConn();

    let svg = mountDiagram(src);
    if (svg) {
      bindSvg(svg);
      return;
    }

    try {
      const response = await fetch(src);
      if (!response.ok) throw new Error("HTTP " + response.status);
      viewer.innerHTML = await response.text();
      svg = viewer.querySelector("svg");
      if (svg) bindSvg(svg);
      else viewer.textContent = "No diagram content found.";
    } catch (error) {
      viewer.textContent = "Failed to load diagram.";
    }
  }

  function closeModal() {
    modal.classList.remove("open");
    viewer.innerHTML = "";
  }

  filters.forEach((filter) => {
    filter.addEventListener("click", () => {
      filters.forEach((item) => item.classList.remove("active"));
      filter.classList.add("active");
      activeType = filter.dataset.type;

      if (activeType === "all") {
        document.querySelectorAll(".entry").forEach((entry) => {
          entry.style.display = "";
        });
        document.querySelectorAll(".architecture-grid").forEach((grid) => {
          grid.style.display = "none";
        });
      } else {
        flattenArchitecture(activeType);
        document.querySelectorAll(".entry").forEach((entry) => {
          entry.style.display = "none";
        });
        document.querySelectorAll(".architecture-grid").forEach((grid) => {
          grid.style.display = "grid";
        });
      }

      filterCards();
    });
  });

  searchInput?.addEventListener("input", filterCards);

  document.addEventListener("click", (event) => {
    const card = event.target.closest(".card");
    if (!card || !card.dataset.src || modal.contains(card)) return;
    const title = card.dataset.title || card.querySelector(".name")?.textContent || "Network";
    openModal(card.dataset.src, title);
  });

  document.addEventListener("keydown", (event) => {
    const card = event.target.closest(".card");
    if (
      card &&
      !modal.contains(card) &&
      (event.key === "Enter" || event.key === " ")
    ) {
      event.preventDefault();
      const title = card.dataset.title || card.querySelector(".name")?.textContent || "Network";
      openModal(card.dataset.src, title);
      return;
    }

    if (event.key === "Escape") closeModal();
    if (!modal.classList.contains("open")) return;
    if (event.key === "+" || event.key === "=") {
      setZoom(Math.min(3, zoom + 0.15));
    }
    if (event.key === "-") {
      setZoom(Math.max(0.4, zoom - 0.15));
    }
  });

  document.getElementById("closeModal")?.addEventListener("click", closeModal);
  modal?.querySelector(".backdrop")?.addEventListener("click", closeModal);
  document.getElementById("zoomIn")?.addEventListener("click", () => setZoom(Math.min(3, zoom + 0.15)));
  document.getElementById("zoomOut")?.addEventListener("click", () => setZoom(Math.max(0.4, zoom - 0.15)));
  document.getElementById("zoomReset")?.addEventListener("click", () => setZoom(1));
  showWeights?.addEventListener("change", () => {
    const svg = viewer.querySelector("svg");
    if (svg) bindSvg(svg);
  });
  showBiases?.addEventListener("change", () => {
    const svg = viewer.querySelector("svg");
    if (svg) bindSvg(svg);
  });

  filterCards();
});
