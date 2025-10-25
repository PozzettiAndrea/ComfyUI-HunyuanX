import { app } from "../../scripts/app.js";

// Register PreviewTrimesh to display interactive 3D viewer in ComfyUI
app.registerExtension({
    name: "MeshCraft.PreviewTrimesh",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PreviewTrimesh") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Get the HTML viewer data
                if (message?.html_viewer && message.html_viewer.length > 0) {
                    const viewerData = message.html_viewer[0];
                    console.log("[PreviewTrimesh] Rendering 3D viewer");
                    console.log(`  Vertices: ${viewerData.mesh_info.vertices}, Faces: ${viewerData.mesh_info.faces}`);

                    // Remove old iframe widget if it exists
                    if (this.widgets) {
                        const oldWidget = this.widgets.find(w => w.name === "3d_viewer_iframe");
                        if (oldWidget) {
                            this.widgets = this.widgets.filter(w => w !== oldWidget);
                        }
                    }

                    // Create iframe widget to display the HTML viewer
                    const widget = this.addCustomWidget({
                        name: "3d_viewer_iframe",
                        type: "HTML_VIEWER",
                        value: viewerData,
                        draw: function(ctx, node, width, y) {
                            // Create or update iframe
                            if (!this.iframe) {
                                this.iframe = document.createElement("iframe");
                                this.iframe.style.width = "100%";
                                this.iframe.style.height = "500px";
                                this.iframe.style.border = "2px solid #444";
                                this.iframe.style.borderRadius = "4px";
                                this.iframe.style.backgroundColor = "white";

                                // Set iframe source to the HTML file
                                this.iframe.src = `/view?filename=preview_trimesh.html&type=temp&subfolder=`;

                                // Find the canvas and insert iframe
                                const canvas = document.querySelector("canvas.graph-canvas");
                                if (canvas && canvas.parentElement) {
                                    canvas.parentElement.appendChild(this.iframe);

                                    // Position iframe over the node
                                    this.updateIframePosition = () => {
                                        const rect = canvas.getBoundingClientRect();
                                        const transform = app.canvas.ds.toScreenSpace(node.pos[0], node.pos[1] + y);
                                        const scale = app.canvas.ds.scale;

                                        this.iframe.style.position = "absolute";
                                        this.iframe.style.left = `${rect.left + transform[0]}px`;
                                        this.iframe.style.top = `${rect.top + transform[1]}px`;
                                        this.iframe.style.width = `${width * scale}px`;
                                        this.iframe.style.height = `${500 * scale}px`;
                                        this.iframe.style.pointerEvents = "auto";
                                        this.iframe.style.zIndex = "1000";
                                    };

                                    this.updateIframePosition();
                                }
                            } else {
                                // Update position if iframe exists
                                if (this.updateIframePosition) {
                                    this.updateIframePosition();
                                }
                            }

                            // Draw placeholder in canvas
                            ctx.fillStyle = "#1e1e1e";
                            ctx.fillRect(0, y, width, 500);
                            ctx.strokeStyle = "#444";
                            ctx.strokeRect(0, y, width, 500);

                            // Draw text overlay
                            ctx.font = "12px monospace";
                            ctx.fillStyle = "#0a0";
                            ctx.fillText("✓ Interactive 3D Viewer", 10, y + 20);
                            ctx.fillStyle = "#888";
                            ctx.fillText(`${this.value.mesh_info.vertices} vertices, ${this.value.mesh_info.faces} faces`, 10, y + 40);
                            ctx.fillText("(Drag to rotate, scroll to zoom)", 10, y + 60);
                        },
                        computeSize: function(width) {
                            return [width, 510];
                        },
                        onRemoved: function() {
                            if (this.iframe && this.iframe.parentElement) {
                                this.iframe.parentElement.removeChild(this.iframe);
                            }
                        }
                    });

                    this.setSize(this.computeSize());

                    // Update iframe position on canvas pan/zoom
                    if (widget.updateIframePosition) {
                        const originalOnMouseMove = app.canvas.onMouseMove;
                        app.canvas.onMouseMove = function(e) {
                            if (originalOnMouseMove) originalOnMouseMove.call(this, e);
                            if (widget.updateIframePosition) widget.updateIframePosition();
                        };
                    }
                }
            };

            // Clean up iframe when node is removed
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                const widget = this.widgets?.find(w => w.name === "3d_viewer_iframe");
                if (widget && widget.iframe && widget.iframe.parentElement) {
                    widget.iframe.parentElement.removeChild(widget.iframe);
                }
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };
        }
    }
});
