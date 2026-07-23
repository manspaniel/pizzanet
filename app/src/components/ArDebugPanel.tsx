import { useState } from "react";
import type {
  ArStatus,
  CaptureRateHz,
  TrackerDebugSettings,
  TrackerFrameWidth,
} from "../ar/types";

const trackerFrameWidthOptions: TrackerFrameWidth[] = [160, 240, 320];
const captureRateOptions: CaptureRateHz[] = [10, 20, 30];

interface ArDebugPanelProps {
  /** Frames come pre-sized and pre-throttled from the native ARKit host, so
   * the resolution and capture-rate controls do not apply. */
  nativeCamera?: boolean;
  onChange: (partial: Partial<TrackerDebugSettings>) => void;
  settings: TrackerDebugSettings;
  status: ArStatus;
}

export function ArDebugPanel({
  nativeCamera = false,
  onChange,
  settings,
  status,
}: ArDebugPanelProps) {
  const [expanded, setExpanded] = useState(import.meta.env.DEV);

  return (
    <aside className="debug-panel" aria-label="Tracker debug panel">
      <button
        type="button"
        className="debug-panel-toggle"
        onClick={() => setExpanded((previous) => !previous)}
      >
        {expanded ? "debug −" : "debug +"}
      </button>

      {expanded && (
        <div className="debug-panel-body">
          <label className="debug-row">
            <span>fov {settings.longAxisFieldOfViewDegrees.toFixed(1)}°</span>
            <input
              type="range"
              min={50}
              max={90}
              step={0.5}
              value={settings.longAxisFieldOfViewDegrees}
              onChange={(event) =>
                onChange({ longAxisFieldOfViewDegrees: Number(event.target.value) })
              }
            />
          </label>
          <label className="debug-row">
            <span>orient delay {settings.visualOrientationDelayMilliseconds}ms</span>
            <input
              type="range"
              min={0}
              max={150}
              step={5}
              value={settings.visualOrientationDelayMilliseconds}
              onChange={(event) =>
                onChange({
                  visualOrientationDelayMilliseconds: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="debug-row">
            <span>features {settings.featureBudget}</span>
            <input
              type="range"
              min={40}
              max={300}
              step={10}
              value={settings.featureBudget}
              onChange={(event) =>
                onChange({ featureBudget: Number(event.target.value) })
              }
            />
          </label>

          <div className="debug-row">
            <label>
              res{" "}
              <select
                disabled={nativeCamera}
                value={settings.trackerFrameWidth}
                onChange={(event) =>
                  onChange({
                    trackerFrameWidth: Number(event.target.value) as TrackerFrameWidth,
                  })
                }
              >
                {trackerFrameWidthOptions.map((width) => (
                  <option key={width} value={width}>
                    {width}px
                  </option>
                ))}
              </select>
            </label>
            <label>
              rate{" "}
              <select
                disabled={nativeCamera}
                value={settings.captureRateHz}
                onChange={(event) =>
                  onChange({
                    captureRateHz: Number(event.target.value) as CaptureRateHz,
                  })
                }
              >
                {captureRateOptions.map((rate) => (
                  <option key={rate} value={rate}>
                    {rate}Hz
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="debug-row">
            <label>
              <input
                type="checkbox"
                checked={settings.pointOverlayEnabled}
                onChange={(event) =>
                  onChange({ pointOverlayEnabled: event.target.checked })
                }
              />{" "}
              points
            </label>
            <label>
              <input
                type="checkbox"
                checked={settings.relocalizationEnabled}
                onChange={(event) =>
                  onChange({ relocalizationEnabled: event.target.checked })
                }
              />{" "}
              reloc
            </label>
            <label>
              <input
                type="checkbox"
                checked={settings.renderSmoothingEnabled}
                onChange={(event) =>
                  onChange({ renderSmoothingEnabled: event.target.checked })
                }
              />{" "}
              smooth
            </label>
          </div>

          <div className="debug-stats">
            <span>{status.frames} frames</span>
            <span>{status.motionSamples} imu</span>
            <span>
              {status.inliers}/{status.matches} vo
            </span>
            <span>{status.relocalizations} reloc</span>
            <span>{Math.round(status.confidence * 100)}% conf</span>
            <span>{status.keyframes} kf</span>
            <span>
              lm {status.convergedLandmarks}/{status.landmarks}
            </span>
            <span>depth {status.meanSceneDepthMetres.toFixed(2)}m</span>
            <span>tex {status.textureScore.toFixed(3)}</span>
            <span>a {status.linearAcceleration.toFixed(2)}</span>
            <span>
              p {status.position.map((value) => value.toFixed(2)).join(" ")}
            </span>
          </div>
        </div>
      )}
    </aside>
  );
}
