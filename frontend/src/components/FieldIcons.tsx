// Field-guide line icons. Stroke uses currentColor.
type IconProps = React.SVGProps<SVGSVGElement> & { size?: number };

function Base({ size = 16, strokeWidth = 1.6, children, ...rest }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" {...rest}>
      {children}
    </svg>
  );
}

export function BirdMark({ size = 22, ...rest }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" {...rest}>
      <path d="M3.2 7.6c2.5-.2 4 .9 4.9 2.2.5-2.2 2.2-4 4.7-4 2.9 0 4.6 2.1 4.9 3.3.2.7 1 .8 1.9.8.7 0 1 .6.5 1.1-.7.7-1.7 1.1-2.8 1-.2 2.6-1.7 5-4.3 6.1l1.2 2.5c.2.5-.1.9-.6.9h-2.2c-.4 0-.7-.2-.8-.6l-.6-1.8c-.5.1-1 .1-1.5.1-3.9 0-6.6-2.6-7.2-6-.3.3-.8.5-1.4.5-.5 0-.8-.6-.4-1 .6-.6.9-1.4.7-2.4-.1-.6.1-1.3.5-1.7.2-.2.4-.5.8-.7Z" />
      <circle cx="13.6" cy="8.1" r="0.9" fill="#fff" />
    </svg>
  );
}
export function SoundIcon({ size = 14, ...rest }: IconProps) {
  return <Base size={size} strokeWidth={1.7} {...rest}><path d="M4 9v6M8 6.5v11M12 9v6M16 7v10M20 10v4" /></Base>;
}
export function ZoomIcon(p: IconProps) { return <Base {...p}><path d="M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM21 21l-4.3-4.3M11 8v6M8 11h6" /></Base>; }
export function CheckIcon(p: IconProps) { return <Base strokeWidth={2} {...p}><path d="M20 6 9 17l-5-5" /></Base>; }
export function BanIcon(p: IconProps) { return <Base {...p}><circle cx="12" cy="12" r="9" /><path d="m5.6 5.6 12.8 12.8" /></Base>; }
export function EditIcon(p: IconProps) { return <Base {...p}><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" /></Base>; }
export function CameraIcon(p: IconProps) { return <Base {...p}><path d="M3 8.5A1.5 1.5 0 0 1 4.5 7H7l1.2-1.8A1 1 0 0 1 9 4.7h6a1 1 0 0 1 .8.5L17 7h2.5A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5Z" /><circle cx="12" cy="12.5" r="3.2" /></Base>; }
export function ChartIcon(p: IconProps) { return <Base {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></Base>; }
export function GearIcon(p: IconProps) { return <Base {...p}><circle cx="12" cy="12" r="3" /><path d="M19.4 13.5a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2v.1a2 2 0 1 1-4 0v-.2a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0-1.2-2.9H3a2 2 0 1 1 0-4h.2a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H10a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.2a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V10a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.2a1.7 1.7 0 0 0-1.4 1.5Z" /></Base>; }
export function FeedIcon(p: IconProps) { return <Base {...p}><path d="M4 5h16M4 12h16M4 19h10" /></Base>; }
export function TagIcon(p: IconProps) { return <Base {...p}><path d="M3 7v5l9 9 7-7-9-9H3Z" /><circle cx="7.5" cy="9.5" r="1.3" /></Base>; }
export function CloseIcon(p: IconProps) { return <Base strokeWidth={1.8} {...p}><path d="M6 6l12 12M18 6 6 18" /></Base>; }
export function SunIcon(p: IconProps) { return <Base {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" /></Base>; }
export function MoonIcon(p: IconProps) { return <Base {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" /></Base>; }
export function ChevronIcon(p: IconProps) { return <Base {...p}><path d="m6 9 6 6 6-6" /></Base>; }
// Indistinct / unidentifiable mark for the Poor Quality sentinel — three
// dashed horizontal strokes evoke a blurred crop the eye can't resolve.
// Distinct from Unknown Bird's "?" (which means "I'm not sure"); this
// means "this can't be known."
export function FogIcon(p: IconProps) { return <Base strokeWidth={1.8} {...p}><path d="M3 7h11M6 12h13M4 17h12" strokeDasharray="2 3" /></Base>; }
