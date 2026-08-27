/**
 * Low-latency CCTV preview: Socket.IO JPEG frames → canvas → MediaStream → <video>.
 *
 * Uses the browser MediaStream API (same transport surface as WebRTC), avoiding
 * MJPEG <img> buffering lag. Falls back to canvas-only when captureStream is unavailable.
 */
(function (global) {
  'use strict';

  const DEFAULT_NS = '/ws/live_video';

  function toArrayBuffer(payload) {
    if (!payload) return null;
    if (payload instanceof ArrayBuffer) return payload;
    if (ArrayBuffer.isView(payload)) {
      return payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength);
    }
    if (payload instanceof Blob) return null; // handled async
    if (typeof payload === 'string') {
      // Rare: base64
      try {
        const bin = atob(payload);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return bytes.buffer;
      } catch (_) {
        return null;
      }
    }
    return null;
  }

  class LivePreviewPlayer {
    /**
     * @param {HTMLVideoElement|HTMLCanvasElement} target
     * @param {{ namespace?: string, mode?: 'raw'|'face', fps?: number, onStatus?: Function }} options
     */
    constructor(target, options) {
      this.target = target;
      this.options = options || {};
      this.namespace = this.options.namespace || DEFAULT_NS;
      this.mode = this.options.mode === 'raw' ? 'raw' : 'face';
      this.quality = String(this.options.quality || 'full').toLowerCase() === 'low' ? 'low' : 'full';
      const low = this.quality === 'low';
      this.fps = Math.max(5, Number(this.options.fps) || (low ? 10 : 15));
      this.onStatus = typeof this.options.onStatus === 'function' ? this.options.onStatus : null;

      this._socket = null;
      this._room = null;
      this._camera = null;
      this._canvas = document.createElement('canvas');
      this._ctx = this._canvas.getContext('2d', { alpha: false });
      this._stream = null;
      this._objectUrl = null;
      this._pending = false;
      this._latestBytes = null;
      this._lastFrameAt = 0;
      this._alive = false;
      this._outW = Math.max(160, Number(this.options.width) || (low ? 480 : 960));
      this._outH = Math.max(90, Number(this.options.height) || (low ? 270 : 540));
      this._canvas.width = this._outW;
      this._canvas.height = this._outH;
    }

    _setStatus(state, detail) {
      if (this.onStatus) {
        try { this.onStatus(state, detail || ''); } catch (_) {}
      }
    }

    async _paintJpeg(bytes) {
      if (!bytes) return;
      // Latest-wins: if a decode is in flight, keep only the newest frame.
      if (this._pending) {
        this._latestBytes = bytes;
        return;
      }
      this._pending = true;
      try {
        let current = bytes;
        while (current) {
          const blob = current instanceof Blob
            ? current
            : new Blob([current], { type: 'image/jpeg' });
          let bitmap = null;
          if (typeof createImageBitmap === 'function') {
            bitmap = await createImageBitmap(blob);
          } else {
            bitmap = await new Promise((resolve, reject) => {
              const img = new Image();
              const url = URL.createObjectURL(blob);
              img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
              };
              img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('jpeg decode failed'));
              };
              img.src = url;
            });
          }
          const w = Math.max(1, bitmap.width || bitmap.naturalWidth || 640);
          const h = Math.max(1, bitmap.height || bitmap.naturalHeight || 360);
          this._ctx.fillStyle = '#000';
          const outW = this._outW;
          const outH = this._outH;
          const scale = Math.min(outW / w, outH / h);
          const dw = w * scale;
          const dh = h * scale;
          const dx = (outW - dw) / 2;
          const dy = (outH - dh) / 2;
          // Draw first, then letterbox bars only — full black clear caused visible blink.
          this._ctx.drawImage(bitmap, dx, dy, dw, dh);
          if (dy > 0) this._ctx.fillRect(0, 0, outW, dy);
          if (dy + dh < outH) this._ctx.fillRect(0, dy + dh, outW, outH - dy - dh);
          if (dx > 0) this._ctx.fillRect(0, dy, dx, dh);
          if (dx + dw < outW) this._ctx.fillRect(dx + dw, dy, outW - dx - dw, dh);
          if (typeof bitmap.close === 'function') bitmap.close();

          this._ensureMediaStream();
          this._lastFrameAt = Date.now();
          this._setStatus('live', `${w}x${h}`);
          current = this._latestBytes;
          this._latestBytes = null;
        }
      } catch (_) {
        this._setStatus('error', 'decode');
      } finally {
        this._pending = false;
        if (this._latestBytes) {
          const leftover = this._latestBytes;
          this._latestBytes = null;
          this._paintJpeg(leftover);
        }
      }
    }

    _ensureMediaStream() {
      const el = this.target;
      if (!el) return;

      // Prefer MediaStream on <video> (WebRTC-compatible sink).
      if (el.tagName === 'VIDEO') {
        const canCapture = typeof this._canvas.captureStream === 'function';
        if (canCapture) {
          if (!this._stream) {
            this._stream = this._canvas.captureStream(this.fps);
            el.srcObject = this._stream;
            el.muted = true;
            el.playsInline = true;
            el.autoplay = true;
            const play = el.play();
            if (play && typeof play.catch === 'function') play.catch(() => {});
          }
          return;
        }
        // Fallback: paint via object URL on video is not ideal; use canvas sibling.
      }

      if (el.tagName === 'CANVAS') {
        const ctx = el.getContext('2d', { alpha: false });
        if (!ctx) return;
        if (el.width !== this._canvas.width || el.height !== this._canvas.height) {
          el.width = this._canvas.width;
          el.height = this._canvas.height;
        }
        ctx.drawImage(this._canvas, 0, 0);
      }
    }

    connect(cameraId) {
      const cam = String(cameraId || '1');
      if (this._alive && this._camera === cam && this._socket && this._socket.connected) {
        this._resubscribe(cam);
        return;
      }
      this.disconnect();
      this._alive = true;
      this._camera = cam;
      this._setStatus('connecting', cam);

      if (typeof io === 'undefined') {
        this._setStatus('error', 'socket.io missing');
        return;
      }

      this._socket = io(this.namespace, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: 800,
      });

      this._socket.on('connect', () => {
        this._resubscribe(cam);
        this._setStatus('connected', cam);
      });

      this._socket.on('subscribed', (info) => {
        this._room = info && info.room ? info.room : null;
      });

      this._socket.on('frame', (payload) => {
        if (!this._alive) return;
        if (payload instanceof Blob) {
          this._paintJpeg(payload);
          return;
        }
        const ab = toArrayBuffer(payload);
        if (ab) this._paintJpeg(ab);
      });

      this._socket.on('disconnect', () => {
        this._setStatus('reconnect', cam);
      });

      this._socket.on('connect_error', () => {
        this._setStatus('error', 'connect');
      });
    }

    _resubscribe(cam) {
      if (!this._socket) return;
      if (this._room) {
        this._socket.emit('unsubscribe', { room: this._room });
        this._room = null;
      }
      this._socket.emit('subscribe', { camera: cam, mode: this.mode, quality: this.quality });
    }

    disconnect() {
      this._alive = false;
      if (this._socket) {
        try {
          if (this._room) this._socket.emit('unsubscribe', { room: this._room });
          this._socket.removeAllListeners();
          this._socket.disconnect();
        } catch (_) {}
      }
      this._socket = null;
      this._room = null;
      this._camera = null;
      if (this.target && this.target.tagName === 'VIDEO') {
        try {
          this.target.srcObject = null;
        } catch (_) {}
      }
      if (this._stream) {
        try {
          this._stream.getTracks().forEach((t) => t.stop());
        } catch (_) {}
      }
      this._stream = null;
      if (this._objectUrl) {
        try { URL.revokeObjectURL(this._objectUrl); } catch (_) {}
        this._objectUrl = null;
      }
      this._setStatus('stopped', '');
    }
  }

  /**
   * Web Audio API alarm for Person of Interest / blacklist matches.
   */
  const AlertAudio = {
    _ctx: null,
    _lastAt: 0,
    cooldownMs: 1800,

    _audioCtx() {
      if (this._ctx) return this._ctx;
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      this._ctx = new AC();
      return this._ctx;
    },

    unlock() {
      const ctx = this._audioCtx();
      if (!ctx) return;
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});
    },

    /**
     * Dual-tone alarm beep (urgent red-alert cadence).
     * @param {'poi'|'blacklist'} kind
     */
    play(kind) {
      const now = Date.now();
      if (now - this._lastAt < this.cooldownMs) return;
      this._lastAt = now;
      const ctx = this._audioCtx();
      if (!ctx) return;
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});

      const isBlacklist = kind === 'blacklist';
      const freqs = isBlacklist ? [880, 660, 880] : [740, 520];
      const t0 = ctx.currentTime;

      freqs.forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.value = freq;
        const start = t0 + i * 0.18;
        const end = start + 0.14;
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(isBlacklist ? 0.22 : 0.14, start + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, end);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(start);
        osc.stop(end + 0.02);
      });
    },
  };

  global.LivePreviewPlayer = LivePreviewPlayer;
  global.AlertAudio = AlertAudio;
})(typeof window !== 'undefined' ? window : globalThis);
