/**
 * Real-time go2rtc WebRTC player (H.264 video, not JPEG frames).
 * Base: /go2rtc (nginx :8088) or http://127.0.0.1:1984
 */
(function (global) {
  'use strict';

  function defaultBases() {
    const forced = (global.GO2RTC_BASE || (typeof localStorage !== 'undefined' && localStorage.getItem('go2rtcBase')) || '').trim();
    if (forced) return [forced.replace(/\/$/, '')];
    // Prefer same-origin Flask proxy, then direct go2rtc.
    return ['/go2rtc', 'http://127.0.0.1:1985', 'http://localhost:1985', 'http://127.0.0.1:1984', 'http://localhost:1984'];
  }

  let _resolvedBase = null;
  let _probePromise = null;

  async function probeBase(base) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 2000);
    try {
      const r = await fetch(`${base}/api/streams`, { signal: ctrl.signal, cache: 'no-store' });
      return r.ok;
    } catch (_) {
      return false;
    } finally {
      clearTimeout(t);
    }
  }

  async function resolveGo2rtcBase(force) {
    if (!force && _resolvedBase) return _resolvedBase;
    if (!force && _probePromise) return _probePromise;
    _probePromise = (async () => {
      for (const base of defaultBases()) {
        if (await probeBase(base)) {
          _resolvedBase = base;
          return base;
        }
      }
      _resolvedBase = null;
      return null;
    })();
    try {
      return await _probePromise;
    } finally {
      _probePromise = null;
    }
  }

  function streamName(cameraId) {
    const id = String(cameraId || '1').replace(/^cam/i, '');
    return `cam${id}`;
  }

  class Go2RtcPlayer {
    constructor(video, options) {
      this.video = video;
      this.options = options || {};
      this.onStatus = typeof this.options.onStatus === 'function' ? this.options.onStatus : null;
      this._pc = null;
      this._camera = null;
      this._alive = false;
      this._base = this.options.base || null;
    }

    _setStatus(state, detail) {
      if (this.onStatus) {
        try { this.onStatus(state, detail || ''); } catch (_) {}
      }
    }

    async connect(cameraId) {
      const cam = String(cameraId || '1');
      this.disconnect();
      this._alive = true;
      this._camera = cam;
      this._setStatus('connecting', cam);

      const base = this._base || (await resolveGo2rtcBase(false));
      if (!base) {
        this._alive = false;
        this._setStatus('error', 'go2rtc offline');
        throw new Error('go2rtc offline — start nvr/start_nvr.bat for real-time WebRTC');
      }
      this._base = base;
      const src = streamName(cam);

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });
      this._pc = pc;
      pc.addTransceiver('video', { direction: 'recvonly' });
      pc.addTransceiver('audio', { direction: 'recvonly' });

      pc.ontrack = (ev) => {
        if (!this._alive || !this.video) return;
        this.video.srcObject = (ev.streams && ev.streams[0])
          ? ev.streams[0]
          : new MediaStream([ev.track]);
        this.video.muted = true;
        this.video.playsInline = true;
        this.video.autoplay = true;
        const p = this.video.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
        this._setStatus('live', src);
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') return resolve();
        const done = () => {
          pc.removeEventListener('icegatheringstatechange', onChange);
          resolve();
        };
        const onChange = () => {
          if (pc.iceGatheringState === 'complete') done();
        };
        pc.addEventListener('icegatheringstatechange', onChange);
        setTimeout(done, 1000);
      });

      if (!this._alive) return;
      const local = pc.localDescription;
      const webrtcUrl = `${base}/api/webrtc?src=${encodeURIComponent(src)}`;

      let answerSdp = null;
      try {
        const r = await fetch(webrtcUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: local.type, sdp: local.sdp }),
        });
        if (r.ok) {
          const ct = (r.headers.get('content-type') || '').toLowerCase();
          if (ct.includes('json')) {
            const j = await r.json();
            answerSdp = j.sdp || null;
          } else {
            answerSdp = await r.text();
          }
        }
      } catch (_) {}

      if (!answerSdp) {
        const r2 = await fetch(webrtcUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: local.sdp,
        });
        if (!r2.ok) throw new Error(`webrtc ${r2.status}`);
        answerSdp = await r2.text();
      }

      if (!this._alive || !answerSdp) throw new Error('empty webrtc answer');
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
      this._setStatus('connected', src);
    }

    disconnect() {
      this._alive = false;
      if (this._pc) {
        try {
          this._pc.getReceivers().forEach((r) => {
            try { r.track && r.track.stop(); } catch (_) {}
          });
          this._pc.close();
        } catch (_) {}
      }
      this._pc = null;
      if (this.video) {
        try { this.video.srcObject = null; } catch (_) {}
      }
      this._setStatus('stopped', '');
    }
  }

  global.Go2RtcPlayer = Go2RtcPlayer;
  global.resolveGo2rtcBase = resolveGo2rtcBase;
  global.go2rtcStreamName = streamName;
})(typeof window !== 'undefined' ? window : globalThis);
